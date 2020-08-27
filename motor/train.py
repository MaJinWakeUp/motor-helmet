# original author: signatrix
# adapted from https://github.com/signatrix/efficientdet/blob/master/train.py
# modified by Zylo117

import datetime
import os
import argparse
import traceback
import shutil

import torch
import yaml
from glob import glob
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from efficientdet.dataset import MotorDataset, generate_sampler
from efficientdet.dataset import Resizer, Normalizer, Augmenter, collater
from backbone import EfficientDetBackbone
from utils_.plotter import Plotter
import numpy as np
from tqdm.autonotebook import tqdm

from efficientdet.loss import FocalLoss
from utils_.sync_batchnorm import patch_replication_callback
from utils_.utils import replace_w_sync_bn, CustomDataParallel, init_weights


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('-p', '--project', type=str, default='motor', help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=2, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=4, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=4, help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=boolean_string, default=False,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--data_path', type=str, default='/home/data/', help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='../../log/')
    parser.add_argument('--graph_path', type=str, default='../../result-graphs/')
    parser.add_argument('-w', '--load_weights', type=str, default='weights/efficientdet-d2.pth',
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='../../models/')
    parser.add_argument('--debug', type=boolean_string, default=False, help='whether visualize the predicted boxes of training, '
                                                                  'the output images will be in test/')

    args = parser.parse_args()
    return args


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss


def train(opt):
    params = Params(f'projects/{opt.project}.yml')

    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    opt.final_path = os.path.join(opt.saved_path, 'final')
    opt.saved_path = os.path.join(opt.saved_path, f'{opt.project}_stages')
    # os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)
    os.makedirs(opt.graph_path, exist_ok=True)
    os.makedirs(opt.final_path, exist_ok=True)

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    training_set = MotorDataset(root_dir=opt.data_path, 
                               transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                             Augmenter(),
                                                             Resizer(input_sizes[opt.compound_coef])]))
    val_set = MotorDataset(root_dir=opt.data_path,
                          transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                        Resizer(input_sizes[opt.compound_coef])]))
    train_sampler, val_sampler = generate_sampler(training_set, 0.9)
    print('Training data: {} , Validation data: {}'.format(len(train_sampler), len(val_sampler)))    
    training_params = {'batch_size': opt.batch_size,
                       'shuffle': (train_sampler is None),
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': opt.num_workers,
                       'sampler': train_sampler}

    val_params = {'batch_size': opt.batch_size,
                  'shuffle': (val_sampler is None),
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': opt.num_workers,
                  'sampler': val_sampler}
    
    training_generator = DataLoader(training_set, **training_params) 
    val_generator = DataLoader(val_set, **val_params)

    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))

    # load last weights
    if opt.load_weights is not None:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}')
    else:
        print('[Info] initializing weights...')
        init_weights(model)

    # freeze backbone if train head_only
    if opt.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if params.num_gpus > 1 and opt.batch_size // params.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

#    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')
    plotter = Plotter(metrics=('loss',), filename = os.path.join(opt.graph_path,
                      f'efficientdet_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.jpg'))

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=opt.debug)

    if params.num_gpus > 0:
        model = model.cuda()
        if params.num_gpus > 1:
            model = CustomDataParallel(model, params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    best_loss = 1e5
    best_epoch = 0
    model.train()

    num_iter_per_epoch = len(training_generator)

    try:
        for epoch in range(opt.start_epoch, opt.num_epochs):
            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                try:
                    imgs = data['img']
                    annot = data['annot']

                    if params.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    optimizer.zero_grad()
                    cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                            epoch+1, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                            reg_loss.item(), loss.item()))


                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue
            scheduler.step(np.mean(epoch_loss))
            plotter.update(epoch+1,'train','loss',np.mean(epoch_loss))

            if (epoch+1) % opt.val_interval == 0:
                model.eval()
                loss_regression_ls = []
                loss_classification_ls = []
                for iter, data in enumerate(val_generator):
                    with torch.no_grad():
                        imgs = data['img']
                        annot = data['annot']

                        if params.num_gpus == 1:
                            imgs = imgs.cuda()
                            annot = annot.cuda()

                        cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss = cls_loss + reg_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss_classification_ls.append(cls_loss.item())
                        loss_regression_ls.append(reg_loss.item())

                cls_loss = np.mean(loss_classification_ls)
                reg_loss = np.mean(loss_regression_ls)
                loss = cls_loss + reg_loss

                print(
                    'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                        epoch+1, opt.num_epochs, cls_loss, reg_loss, loss))
                plotter.update(epoch+1,'val','loss',loss)

                
                if loss + opt.es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch

                model.train()
                           
                # Early stopping
                if epoch - best_epoch > opt.es_patience > 0:
                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch+1, best_loss))
                    break
            if epoch==0 or (epoch+1)%10==0:
                save_checkpoint(model, 'efficientdet-d{}_{}.pth'.format(
                                    opt.compound_coef, epoch+1))              
            plotter.draw_curve()
                
    except KeyboardInterrupt:
        save_checkpoint(model, 'efficientdet-d{}_{}.pth'.format(
                                opt.compound_coef, epoch+1))
    if opt.final_path and os.path.exists(opt.final_path):
        move_final_best(opt.saved_path, opt.final_path)
        
def move_final_best(src_dir, dst_dir):
    weights_path = glob(src_dir + f'/*.pth')
    weights_path = sorted(weights_path,
                          key=lambda x: int(x.rsplit('_')[-1].split('.')[0]),
                          reverse=True)[0]

    print(f'using weights {weights_path} as final best weights.')
    shutil.copy(weights_path, os.path.join(dst_dir,f'efficientdet-d{opt.compound_coef}.pth'))

def save_checkpoint(model, name):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(opt.saved_path, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(opt.saved_path, name))


if __name__ == '__main__':
    opt = get_args()
    torch.cuda.empty_cache()
    train(opt)
