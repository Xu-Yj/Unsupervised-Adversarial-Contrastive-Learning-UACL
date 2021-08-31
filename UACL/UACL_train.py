from PIL import Image
import torch
from tqdm import tqdm
from torchvision import models
from torch import nn
from torch.utils.data import DataLoader
from UACL import UACL
from torchvision import transforms
import argparse

parser = argparse.ArgumentParser(description='Pytorch MSTAR Training')
parser.add_argument('--data',type=str,default='train_list.txt',help='the address of txt that contains the information of data')
parser.add_argument('--if_al',type=bool,default=True,help='if the adversarial learning is used in this self-supervised learning,you can set False to pre-train a Siamese network for further training')
parser.add_argument('--image_size',type=int,default=128,help='image size')
parser.add_argument('--batch_size',type=int,default=8,help='batch size')
parser.add_argument('--class_num',type=int,default=10,help='class num of data')
parser.add_argument('--epochs',type=int,default=200,help='number of total epochs to run')
parser.add_argument('--checkpoint',type=str,default='checkpoints',help='address of saving pretrained model')
parser.add_argument('--save_frep',type=int,default=100,help='save frequency')
parser.add_argument('--attack_strength',type=int,default=8,help='attack strength of unsupervised attack')
parser.add_argument('--attack_step_size',type=int,default=2,help='attack step size of unsupervised attack')
parser.add_argument('--attack_iter',type=int,default=10,help='attack iter of unsupervised attack')
parser.add_argument('--lr',type=float,default=5.0,help='optimizer lr')
parser.add_argument('--weight_decay',type=float,default=1e-6,help='optimizer weight decay')
parser.add_argument('--momentum',type=float,default=0.9,help='optimizer momentum')
parser.add_argument('--mean',type=float,default=0.184,help='mean of dataset')
parser.add_argument('--std',type=float,default=0.119,help='standard deviation of dataset')
parser.add_argument('--shuffle',type=bool,default=False,help='if the dataset is random shuffled')
parser.add_argument('--seed',type=int,default=1,help='random seed')


global args
args = parser.parse_args()
class MyDataset(torch.utils.data.Dataset): 
    def __init__(self, root, datatxt1, datatxt2, transform=None, target_transform=None): 
        super(MyDataset, self).__init__()

        fh1 = open(root + datatxt1, 'r') 
        imgs1 = [] 
        for line in fh1:  
            line = line.rstrip()  
            words = line.split() 
            imgs1.append((words[0], int(words[1])))  
            
        self.imgs1 = imgs1

        fh2 = open(root + datatxt2, 'r') 
        imgs2 = []  
        for line in fh2:  
            line = line.rstrip() 
            words = line.split() 
            imgs2.append((words[0], int(words[1])))  
            
        self.imgs2 = imgs2

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):  
        fn1, label1 = self.imgs1[index]  

        img1 = Image.open(fn1).convert('RGB')  

        if self.transform is not None:
            img1 = self.transform(img1) 
        fn2, label2 = self.imgs2[index]  
        img2 = Image.open(fn2).convert('RGB')  

        if self.transform is not None:
            img2 = self.transform(img2)  
        return img1, img2  
    def __len__(self):  
        return len(self.imgs1)

mean = args.mean
std = args.std
class_num = args.class_num
image_size = args.image_size
if_al=args.if_al
transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((mean,),(std,))])  # 非resnet101



root = 'txts/'
ds = MyDataset(root=root, datatxt1=args.data, datatxt2=args.data, transform=transform)
train_loader = torch.utils.data.DataLoader(ds,
                                           batch_size=args.batch_size, shuffle=args.shuffle,
                                           num_workers=4)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(args.seed)
resnet=models.resnet18()
resnet.fc=nn.Linear(in_features=512, out_features=class_num, bias=True)
uacl = UACL(resnet,
                   image_size=image_size,
                   hidden_layer='avgpool')
uacl = uacl.cuda()
opt = torch.optim.SGD(uacl.parameters(),lr=args.lr,weight_decay=args.weight_decay,momentum=args.momentum)
def project(x, original_x, epsilon, _type='linf'):
    if _type == 'linf':
        max_x = original_x + epsilon
        min_x = original_x - epsilon
        x = torch.max(torch.min(x, max_x), min_x)
    else:
        raise NotImplementedError
    return x
def clip_by_tensor(t,t_min,t_max):
    m1=t>t_min
    m2=~m1
    t_mew=m1*t+m2*t_min
    n1=t<t_max
    n2=~n1
    t_mew=n1*t_mew+n2*t_max
    return t_mew

if __name__ == '__main__':
    attack_strength = args.attack_strength
    step_size = args.attack_step_size
    attack_iter = args.attack_iter
    save_frep = args.save_frep
    epochs = args.epochs
    checkpoint = args.checkpoint
    with torch.enable_grad():
        for i in tqdm(range(epochs)):
            for data in train_loader:
                inputs1, inputs2 = data
                inputs1 = inputs1.cuda()
                inputs2 =inputs2.cuda()
                if if_al:
                    inputs2.requires_grad = True
                    inputs2.requires_grad_()
                    pertubation=torch.zeros(inputs2.shape).type_as(inputs2).cuda()
                    min,max=inputs2 - attack_strength / 255 / std, inputs2 + attack_strength / 255 / std
                  
                    #unsupervised adversarial attack
                    for _ in range(attack_iter):
                        loss = uacl(inputs1, inputs2)
                        grad_outputs = None
                        grads = torch.autograd.grad(loss, inputs2, grad_outputs=grad_outputs, only_inputs=True, retain_graph=False)[0]
                        pertubation=step_size/255/std*torch.sign(grads)

                        inputs2=clip_by_tensor(inputs2+pertubation,min,max)
                        inputs2=clip_by_tensor(inputs2,-mean/std, (1-mean)/std)
                loss = uacl(inputs1, inputs2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                uacl.update_moving_average()

                loss = uacl(inputs1, inputs2)
            if (i+1)%save_frep == 0:
                torch.save(resnet,
                           '{}/UACL_encoder_{}epoch.pkl'.format(checkpoint,i+1))
                torch.save(uacl,
                           '{}/UACL_Siamese_network_{}epoch.pkl'.format(checkpoint,i+1))
        torch.save(resnet,
                   '{}/UACL_encoder.pkl'.format(checkpoint))
        torch.save(uacl,
                   '{}/UACL_Siamese_network.pkl'.format(checkpoint))
