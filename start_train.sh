#!/bin/bash

project_root_dir=/project/train/src_repo
dataset_dir=/home/data
preproc_dir=/project/train/processed_data
graph_dir=/project/train/result-graphs
model_dir=/project/train/models

echo "Install requirements..." 
pip install -i https://mirrors.aliyun.com/pypi/simple -r /project/train/src_repo/requirements.txt 

echo "Computing anchors for motor detection..." 
cd ${project_root_dir}/motor/utils_
python3 anchors_motor.py --data_path=${dataset_dir} --yaml_file=/project/train/src_repo/motor/projects/motor.yml

echo "Start training model for motor detection..." 
cd ${project_root_dir}/motor
python3 -u train.py -c=2 -w='/project/train/src_repo/motor/weights/efficientdet-d2.pth' --optim='adamw' --num_workers=0 --batch_size=4 --num_epochs=100 --start_epoch=0 --es_patience=20 --data_path=${dataset_dir} --graph_path=${graph_dir} --saved_path=${model_dir}
python3 -u train.py -c=2 -w='/project/train/models/final/efficientdet-d2.pth' --optim='sgd' --num_workers=0 --batch_size=4 --num_epochs=150 --start_epoch=100 --es_patience=11 --data_path=${dataset_dir} --graph_path=${graph_dir} --saved_path=${model_dir}
rm -rf ${model_dir}/motor_stages

echo "Start processing images for head and helmet detection..." 
cd ${project_root_dir}/motor/utils_
python3 crop_motor.py --data_path=${dataset_dir} --save_path=${preproc_dir}

cd ${project_root_dir}/head
python3 -u train.py --num_workers=0 --data_path=${preproc_dir} --graph_path=${graph_dir} --saved_path=${model_dir} --es_patience=30