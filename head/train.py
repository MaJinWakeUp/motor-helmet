from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import argparse
import torch.utils.data as data
from data import HeadHelmetDataset, detection_collate, preproc, cfg_mnet, cfg_re50, generate_sampler
from layers.modules import MultiBoxLoss, FocalLoss
from layers.functions.prior_box import PriorBox
from utils.plotter import Plotter
import time
import datetime
import math
from models.retinaface import RetinaFace
from tqdm.autonotebook import tqdm
import shutil
from glob import glob


parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--data_path', default='/home/data/', help='Training dataset directory')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--saved_path', default='../../models/', help='Location to save checkpoint models')
parser.add_argument('--graph_path', type=str, default='../../result-graphs/')
parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
parser.add_argument('--es_min_delta', type=float, default=0.0, help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
parser.add_argument('--es_patience', type=int, default=0, help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')

args = parser.parse_args()

graph_path = args.graph_path
final_path = os.path.join(args.saved_path, 'final')
saved_path = os.path.join(args.saved_path, 'head_stages')
os.makedirs(saved_path, exist_ok=True)
os.makedirs(graph_path, exist_ok=True)
os.makedirs(final_path, exist_ok=True)
    
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50

rgb_mean = (104, 117, 123) # bgr order
classes = cfg['classes']
num_classes = len(classes)
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
data_path = args.data_path

val_interval = args.val_interval
es_min_delta = args.es_min_delta
es_patience = args.es_patience

net = RetinaFace(cfg=cfg)
print("Printing net...")
print(net)

if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()

cudnn.benchmark = True


optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)
#criterion = FocalLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()

def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = HeadHelmetDataset(data_path, preproc(img_dim, rgb_mean))
    train_sampler, val_sampler = generate_sampler(dataset, 0.9)
    training_generator = data.DataLoader(dataset, batch_size, shuffle=False, sampler=train_sampler,
                                         num_workers=num_workers, collate_fn=detection_collate) 
    val_generator = data.DataLoader(dataset, batch_size, shuffle=False, sampler=val_sampler,
                                         num_workers=num_workers, collate_fn=detection_collate) 
    print('Training data: {} , Validation data: {}'.format(len(train_sampler), len(val_sampler)))    
    plotter = Plotter(phases=('train','val'), metrics=('loss',), filename = os.path.join(graph_path,
                      f'retinaface_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.jpg'))

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

#    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
#    step_index = 0
    

    if args.resume_epoch > 0:
        start_epoch = args.resume_epoch
    else:
        start_epoch = 0
    try:
        best_loss = 1e5
        best_epoch = 0
        for epoch in range(start_epoch, max_epoch):
            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for i, (images, targets) in enumerate(progress_bar):
                step = epoch * epoch_size + i
                load_t0 = time.time()
#                if step in stepvalues:
#                    step_index += 1
#                lr = adjust_learning_rate(optimizer, gamma, epoch+1, step_index, step, epoch_size)
                images = images.cuda()
                targets = [anno.cuda() for anno in targets]
                
                out = net(images)
                
                optimizer.zero_grad()
                loss_l, loss_c = criterion(out, priors, targets)
                loss = cfg['loc_weight'] * loss_l + cfg['cls_weight'] * loss_c
                epoch_loss.append(float(loss))
                loss.backward()
                optimizer.step()
                load_t1 = time.time()
                batch_time = load_t1 - load_t0
                eta = int(batch_time * (max_iter - step))
                progress_bar.set_description('Epoch:{}/{} || Epochiter: {}/{} || Loc: {:.4f} Cla: {:.4f} || Batchtime: {:.4f} s || ETA: {}'
                  .format(epoch+1, max_epoch, i+1, epoch_size, loss_l.item(), loss_c.item(), batch_time, str(datetime.timedelta(seconds=eta))))
            plotter.update(epoch+1,'train','loss',np.mean(epoch_loss))
            plotter.draw_curve()
            scheduler.step(np.mean(epoch_loss))
            if epoch==0 or (epoch+1) % 20 == 0 or ((epoch+1) % 5 == 0 and epoch > cfg['decay2']):
                save_checkpoint(net, f'RetinaFace_{epoch+1}.pth')
            
            
            if (epoch+1) % val_interval == 0:
                net.eval()
                losses = []
                for i, (images, targets) in enumerate(val_generator):
                    images = images.cuda()
                    targets = [anno.cuda() for anno in targets]
                    out = net(images)
                    loss_l, loss_c = criterion(out, priors, targets)
                    loss_ = cfg['loc_weight'] * loss_l + cfg['cls_weight'] * loss_c
                    losses.append(float(loss_))
                loss = np.mean(losses)
                print('Val. Epoch: {}/{}. Total loss: {:.4f}'.format(
                                epoch+1, max_epoch, loss))
                plotter.update(epoch+1,'val','loss',loss)
    
                    
                if loss + es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch
                # Early stopping
                if epoch - best_epoch > es_patience > 0:
                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch+1, best_loss))
                    break
                net.train()
                               

                
    except KeyboardInterrupt:
        save_checkpoint(net, f'RetinaFace_{epoch+1}.pth')
    if final_path and os.path.exists(final_path):
        move_final_best(saved_path, final_path)

    
def move_final_best(src_dir, dst_dir):
    weights_path = glob(src_dir + f'/*.pth')
    weights_path = sorted(weights_path,
                          key=lambda x: int(x.rsplit('_')[-1].split('.')[0]),
                          reverse=True)[0]
    print(f'using weights {weights_path} as final best weights.')
    shutil.copy(weights_path, os.path.join(dst_dir,'RetinaFace.pth'))


def save_checkpoint(model, name):
    torch.save(model.state_dict(), os.path.join(saved_path, name))

def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    torch.cuda.empty_cache()
    train()
