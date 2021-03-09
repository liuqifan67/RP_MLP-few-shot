# RP_MLP-few-shot
  Prerequisites
  
  Python >= 3.6
  
  Pytorch >= 1.4 and torchvision (https://pytorch.org/)
  
# step
1. train feature encoder

 python3 train_baseline.py --method PRETRAIN --dataset miniImagenet --name PRETRAIN --train_aug
 
2. train metric network

python3 train_baseline.py --method GNN --dataset cub --testset TESTSET --name metric_TESTSET_METHOD --warmup PRETRAIN --train_aug
dataset：miniImagenet, cars, cub, places, or plantae 
