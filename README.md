# Unsupervised-Adversarial-Contrastive-Learning-UACL
This repository is the official PyTorch implementation of "Unsupervised Adversarial Contrastive Learning" by Yanjie Xu and Hao Sun.
## Requirements
Currently, requires following packages
+ python 3.6+
+ torch 1.6+
+ torchvision 0.7+
+ CUDA 10.1+
+ tqdm 4.36+
+ PIL 6.2.2+
## Training
To train the model(s) in the paper, run this command:
```
python -m  UACL_train.py --data='train_list.txt' --batch_size=8 --image_size=128 --class_num=10 --epochs=200 --checkpoint='checkpoint' --save_frep=50 --attack_strength=8 --attack_step_size=8 --attack_iter=10  --lr=5.0 --weight_decay=1e-6 --momentum =0.9 --mean=0.184 --std=0.119 --seed=1
```

## Finetune
To finetune the pretrained model, run this command:
```
python -m  finetune.py --data_dir='MSTAR' --pretrained_model_dir='checkpoint/UACL_encoder.pkl' --checkpoint='checkpoints' --image_size=128 --batch_size=8 --epochs=100 --mean=0.184 --std=0.119
```
