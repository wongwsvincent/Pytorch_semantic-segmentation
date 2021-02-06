import time
import os
from random import randrange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from loss_entropy2d import CrossEntropy2d, CrossEntropyLoss2d

import torch
from torch import nn
from torch.autograd import Variable 
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Sampler, DataLoader
from torchvision.transforms import transforms

from models.resnet.resnet import resnet56
from models.senet.se_resnet import se_resnet56
from models.pspnet.pspnet import pspnet50
from models.unet.unet_model import unet

from losses.losses import CrossEntropyLoss2d


parser = argparse.ArgumentParser(description="This python script is used to check the maximum pixel size of glomerulus, and crop around the glomerulus with the desired pixel size.")
parser.add_argument("-m", "--model",
                    action="store",
                    dest="model_type",
                    default="pspnet50", # 'resnet56', 'se_resnet56', 'pspnet50', 'unet'
                    type=str,
                    help="the choice of model")
parser.add_argument("-l", "--loss",
                    action="store",
                    dest="loss_type",
                    default="CrossEntropy", # 'CrossEntropy', 'Dice', 'Jaccard'
                    type=str,
                    help="the choice of loss function")
parser.add_argument("-p", "--path",
                    dest="root_dir",
                    default="",
                    help="the (root) directory in which the dataset is stored",
                    required=True)
parser.add_argument("-g", "--gpu",
                    dest="gpu",
                    default="",
                    help="the GPU device number")
parser.add_argument("-d", "--device",
                    dest="device",
                    default="gpu",
                    help="By default it tries to use GPU. Set to 'cpu' to force using CPU instead.")
parser.add_argument("-cv", "--cross_validation",
                    action="store_true",
                    dest="cv_flg",
                    default=False,
                    help="Perform cross validation.")
parser.add_argument("-lr", "--learning_rate",
                    action="store",
                    dest="learning_rate",
                    default=0.00001,
                    type=float,
                    help="setting the learning rate")
parser.add_argument("-e", "--num_epoch",
                    action="store",
                    dest="numEpoch",
                    default=60,
                    type=int,
                    help="setting the number of epoch")


args = parser.parse_args()

if( not os.path.isdir(args.root_dir) ):
    sys.exit("Error: No path found!")
elif( not os.path.isdir(os.path.join(args.root_dir,"processed")) ):
    sys.exit("Error: Check your path!")


### set which GPU device to use
if(args.gpu=="0" or args.gpu=="1"):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
elif(args.gpu!=""):
    sys.exit("Error: use 0 or 1 to specify your GPU device")

if(args.device=="gpu"):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using GPU? {}".format(torch.cuda.is_available()))
elif(args.device=="cpu"):
    device = "cpu"
    print("Using GPU? False")
else:
    sys.exit("Error: use gpu or cpu for the device option.")


### split the data samples and splits into folds
df_train = pd.DataFrame(columns=["img_name","label"])
df_train["img_name"] = os.listdir(os.path.join(args.root_dir,"processed/train"))
list_labels=[]
for idx, filename in enumerate(df_train["img_name"]):
    if "glomerulus_" in filename:
        list_labels.append(1)
    if "medulla_" in filename:
        list_labels.append(2)
    if "cortex_" in filename:
        list_labels.append(3)
df_train["label"] = list_labels

list_img_names=os.listdir(os.path.join(args.root_dir,"processed/train"))
splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
splits = []
for train_idx, test_idx in splitter.split(df_train['img_name'].to_numpy(), df_train['label'].to_numpy()):
    splits.append((train_idx, test_idx))

### Set up dataset and dataloader classes
class HuBMAPDataset(torch.utils.data.Dataset):
    def __init__(self, img_names, transform=[transforms.ToTensor()], device='cpu'):
        self.img_names = img_names
        self.transform = transforms.Compose(transform)
        self.device=device
    def __len__(self):
        return len(self.img_names)
    def __getitem__(self, img_name):
        img=Image.open(os.path.join(args.root_dir,"processed/train",img_name)).convert("RGB")
        if self.transform is not None:
            img=self.transform(img).to(device)
        mask=transforms.Compose([transforms.ToTensor()])(Image.open(os.path.join(args.root_dir,"processed/mask",img_name)).convert("1")).to(device)
        return (img, mask)

class ImageSampler(Sampler):
    def __init__(self, sample_idx):
        self.sample_idx = sample_idx
        self.df_images=pd.DataFrame(os.listdir(os.path.join(args.root_dir,"processed/train")), columns=['img_name'])
    def __iter__(self):
        image_ids = self.df_images['img_name'].loc[self.sample_idx]
        return iter(image_ids)
    def __len__(self):
        return len(self.sample_idx)
    
def create_split_loader(dataset, split, batch_size):
    train_folds_idx = split[0]
    valid_folds_idx = split[1]
    train_sampler = ImageSampler(train_folds_idx)
    valid_sampler = ImageSampler(valid_folds_idx)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    return (train_loader, valid_loader)    
    
def get_all_split_loaders(dataset, cv_splits, batch_size=30):
    """Create DataLoaders for each split.
    Keyword arguments:
    dataset -- Dataset to sample from.
    cv_splits -- Array containing indices of samples to 
                 be used in each fold for each split.
    batch_size -- batch size.
    """
    split_loaders = []
    for i in range(len(cv_splits)):
        split_loaders.append(
            create_split_loader(dataset, cv_splits[i], batch_size)
        )
    return split_loaders

dataset = HuBMAPDataset(df_train['img_name'].to_numpy(), device=device)
if(args.cv_flg==True):
    if(args.model_type=='resnet56'):
        dataloaders = get_all_split_loaders(dataset, splits, batch_size=3)
    elif(args.model_type=='se_resnet56'):
        dataloaders = get_all_split_loaders(dataset, splits, batch_size=2)
    elif(args.model_type=='pspnet50'):
        dataloaders = get_all_split_loaders(dataset, splits, batch_size=2)
    elif(args.model_type=='unet'):
        dataloaders = get_all_split_loaders(dataset, splits, batch_size=2)
else:
    dataloaders = DataLoader(dataset, batch_size=2)


if(args.model_type=='resnet56'):
    model=resnet56().to(device)
elif(args.model_type=='se_resnet56'):
    model=se_resnet56().to(device)
elif(args.model_type=='pspnet50'):
    model=pspnet50().to(device)
elif(args.model_type=='unet'):
    model=unet().to(device)

torch.save(model.state_dict(), 'myWeight_{}_init.pt'.format(args.model_type))
m_state_dict = torch.load('myWeight_{}_init.pt'.format(args.model_type))

if(loss_choice=='CrossEntropy'):
    criterion = CrossEntropyLoss2d() # This criterion combines LogSoftMax and NLLLoss in one single class

optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)


start_time = time.time()

torch.cuda.empty_cache()
list_trainloss=[]
list_testloss=[]

numEpoch=args.numEpoch
for i, (train_batch_loader, valid_batch_loader) in enumerate(dataloaders, 0):
    print("In cross-validation set {}:".format(i+1))
    model.load_state_dict(m_state_dict)
    tmp_list_trainloss_perEpoch=[]
    tmp_list_testloss_perEpoch=[]
    for epoch in range(numEpoch):
        tmp_trainloss=0
        tmp_testloss=0
        tmp_trainloss_cnt=0
        tmp_testloss_cnt=0
        # Loop through all batches in training folds for a given split.
        for j, (images, masks) in enumerate(train_batch_loader,0):
            y_pred=model(images)
            loss=criterion(y_pred,masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tmp_trainloss+=loss.item()
            tmp_trainloss_cnt+=1
            del images
            del masks
            del y_pred
            del loss
            if ((j+1)%100==0):
                print("Training batch {}...".format(j+1))
        print("Done training")
        
        # Loop through all batches in validation fold for a given split.
        with torch.no_grad():
            model.eval()
            for j, (images, masks) in enumerate(valid_batch_loader,0):
                y_pred=model(images)
                loss=criterion(y_pred,masks)
                tmp_testloss+=loss.item()
                tmp_testloss_cnt+=1
                del images
                del masks
                del y_pred
                del loss
                if ((j+1)%100==0):
                    print("Testing batch {}...".format(j+1))
        model.train()
        print("Done testing")

        tmp_list_trainloss_perEpoch.append(tmp_trainloss/tmp_trainloss_cnt)
        tmp_list_testloss_perEpoch.append(tmp_testloss/tmp_testloss_cnt)
        print("Finished epoch {}...".format(epoch+1))
        print()
        
    torch.save(model.state_dict(), 'myWeight_{}_cv{}.pt'.format(model_choice,i))
    list_trainloss.append(tmp_list_trainloss_perEpoch)
    list_testloss.append(tmp_list_testloss_perEpoch)
    del tmp_list_trainloss_perEpoch
    del tmp_list_testloss_perEpoch
    print("Finished all epochs in cross-validation set {}".format(i+1))
    print()

### TEST: plotting some images from the first batch
# tensor_to_image = transforms.ToPILImage()
# fig, ax = plt.subplots(2,5, figsize=(20, 20))
# images = train_batch[0]
# masks = train_batch[1]
# for i,image in enumerate(images):
#     if i < 5:
#         tmp_img=tensor_to_image(image.cpu())
#         print(tmp_img)
#         ax[0,i].imshow(tmp_img)
# for i,mask in enumerate(masks):
#     if i < 5:
#         tmp_mask=tensor_to_image(mask.cpu())
#         print(tmp_mask)
#         ax[1,i].imshow(tmp_mask)
# plt.show()

print("--- %s minutes ---" % ((time.time() - start_time)/60))
