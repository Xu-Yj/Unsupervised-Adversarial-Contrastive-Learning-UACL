from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import time
import os
from tqdm import trange

import argparse

parser = argparse.ArgumentParser(description='Pytorch Finetune')
parser.add_argument('--data_dir',type=str,default='MSTAR',help='the address of data')
parser.add_argument('--pretrained_model_dir',type=str,default='MSTAR',help='the address of data')
parser.add_argument('--checkpoint',type=str,default='checkpoints',help='address of saving pretrained model')
parser.add_argument('--image_size',type=int,default=128,help='image size')
parser.add_argument('--batch_size',type=int,default=8,help='batch size')
parser.add_argument('--epochs',type=int,default=1,help='number of total epochs to run')
parser.add_argument('--checkpoint',type=str,default='checkpoints',help='address of saving pretrained model')
parser.add_argument('--mean',type=float,default=0.184,help='mean of dataset')
parser.add_argument('--std',type=float,default=0.119,help='standard deviation of dataset')
parser.add_argument('--shuffle',type=bool,default=True,help='if the dataset is random shuffled')
parser.add_argument('--batch_size',type=int,default=8,help='batch size')

global args
args = parser.parse_args()
mean = args.mean
std = args.std
image_size = args.image_size
batch_size = args.batch_size
shuffle = args.shuffle
data_dir = args.data_dir
pretrained_model_dir = args.pretrained_model_dir
checkpoint = args.checkpoint

def train_model(model, criterion, optimizer, scheduler, num_epochs=100):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in trange(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs ))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
                model.eval()
            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for data in dataloders[phase]:
                inputs, labels = data
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                # print (outputs.shape)
                _, preds = torch.max(outputs.data, 1)
                # print (outputs.shape)
                # print (preds.shape)
                loss = criterion(outputs, labels)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).to(torch.float32)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(1),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([mean, ], [std, ])
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(1),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([mean, ], [std, ])
        ])
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x]) for x in ['train', 'val']}
    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=4) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    # use gpu or not
    use_gpu = torch.cuda.is_available()
    # get model and replace the original fc layer with your fc layer
    model_ft = torch.load(pretrained_model_dir)
    if use_gpu:
        model_ft = model_ft.cuda()
    # define loss function
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=11, gamma=0.9)
    model_ft = train_model(model=model_ft,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=batch_size)
    torch.save(model_ft,"{}/UACL_classifier.pth".format(checkpoint))