# UACL-Unsupervised Adversarial Contrastive Learning
This repository is the official PyTorch implementation of "Unsupervised Adversarial Contrastive Learning" by Yanjie Xu, Hao Sun, Jin Chen, Lin Lei, Kefeng Ji and Gangyao Kuang.
## Requirements
Currently, requires following packages
* python 3.6+
* torch 1.6+
* torchvision 0.7+
* CUDA 10.1+
* tqdm 4.41+
* PIL 6.2+
* kornia 0.4+
## Training
To train the model(s) in the paper, run this command:
```
python -m  UACL_train --data='train_list.txt' --batch_size=8 --image_size=128 --class_num=10 --epochs=200 --checkpoint='checkpoints' --save_frep=50 --attack_strength=8 --attack_step_size=8 --attack_iter=10  --lr=5.0 --weight_decay=1e-6 --momentum=0.9 --mean=0.184 --std=0.119 --seed=1
```
To finetune the pretrained model(s) in the paper, run this command:
```
python -m  finetune --data_dir='MSTAR' --pretrained_model_dir='checkpoints/UACL_encoder.pkl' --checkpoint='checkpoints' --image_size=128 --batch_size=8 --epochs=100 --mean=0.184 --std=0.119
```

