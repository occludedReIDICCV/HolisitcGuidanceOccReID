from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import time
import os
import scipy.io
from model import  PCB, ClassBlock
from PIL import Image
from shared_region_evaluate import evaluate
from utils.part_label import part_label_generate
######################################################################
# Options
# --------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRID_SPACING = 10
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--which_epoch',default='59', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='./dataset/Occluded_Duke/processed_data',type=str, help='./test_data')
parser.add_argument('--result_dir',default='./result',type=str, help='result path')
parser.add_argument('--name', default='PGFA', type=str, help='save model path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--gallery_heatmapdir',default='./heatmaps/18heatmap_gallery',type=str, help='gallery heatmap path')
parser.add_argument('--query_heatmapdir',default='./heatmaps/18heatmap_query',type=str, help='query heatmap path')
parser.add_argument('--gallery_posedir',default='./test_pose_storage/gallery/sep-json',type=str, help='gallery pose path')
parser.add_argument('--query_posedir',default='./test_pose_storage/query/sep-json',type=str, help='query pose path')

parser.add_argument('--train_classnum', default=702, type=int, help='train set class number')
parser.add_argument('--part_num', default=3, type=int, help='part_num')
parser.add_argument('--hidden_dim', default=256, type=int, help='hidden_dim')
opt = parser.parse_args()

name = opt.name
test_dir = opt.test_dir


# set gpu ids
data_dir = test_dir
###
######################################################################
# Load Data
# ---------
data_transforms = transforms.Compose([
        transforms.Resize((384,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


######################################################################
class BaseDataset(Dataset):
    def __init__(self,test_datapath,mask_path):
        self.datapath=test_datapath
        self.transform=data_transforms
        self.mask_path=mask_path
        self.ids = sorted(os.listdir(self.datapath))
        self.classnum=len(self.ids)
        self.data=[]
        for pid in sorted(self.ids):
            for img in os.listdir(os.path.join(self.datapath,pid)):
                imgpath = os.path.join(self.datapath,pid,img)
                cam_id = int(img.split('c')[1][0])
                self.data.append((imgpath,int(pid),int(cam_id),img))
    def __getitem__(self,index):
        imgpath,pid,cam_id,imgname=self.data[index]
        img=Image.open(imgpath).convert('RGB')
        w,h = img.size
        img = self.transform(img)
        mask_name = imgpath.split('/')[-1].split('.')[0]+'.npy'
        mask=np.load(os.path.join(self.mask_path,mask_name))
        mask=torch.from_numpy(mask)
        mask=mask.float()
        return img,pid,cam_id,mask,imgname,h
    def __len__(self):
        return len(self.data)

######################################################################
# Load model
#---------------------------

def load_network(name_,network):
    save_path = os.path.join(opt.result_dir,name,'%s_%s.pth'%(name_,opt.which_epoch))
    network.load_state_dict(torch.load(save_path))
    return network

#########################################################################
def extract_global_feature(model,input_):
    output=model(input_)
    return output
###
def extract_partial_feature(model,global_feature,part_num):
    partial_feature=nn.AdaptiveAvgPool2d((part_num,1))(global_feature)
    partial_feature=torch.squeeze(partial_feature,-1)
    partial_feature=partial_feature.permute(0,2,1)
    return partial_feature
###
def extract_pg_global_feature(model,global_feature,masks):
    pg_global_feature_1=nn.AdaptiveAvgPool2d((1,1))(global_feature)
    pg_global_feature_1=pg_global_feature_1.view(-1,2048)
    pg_global_feature_2=[]
    for i in range(18):
        mask=masks[:,i,:,:]
        mask= torch.unsqueeze(mask,1)
        mask=mask.expand_as(global_feature)
        pg_feature_=mask*global_feature
      
        pg_global_feature_2.append(pg_feature_)
    pg_global_feature_2=torch.cat((pg_global_feature_2),2)
   
    return pg_global_feature_2

###
def feature_extractor(data_path,mask_path,pose_path,model,partial_model,global_model):
    total_part_label=[]

    total_partial_feature=[]  #storage of partial feature 
    total_pg_global_feature=[] #storage of pose guided global feature
    total_label=[] #storage of gallery label
    total_cam=[] #storage of gallery cam
    list_img=[]

    image_dataset = BaseDataset(data_path,mask_path)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=opt.batchsize,
                                                 shuffle=False,num_workers=0)
    for it, data in enumerate(dataloader):
        imgs,pids,cam_ids,masks,imgnames,heights=data
        imgs= imgs.cuda()
        masks=masks.cuda()
        global_feature=extract_global_feature(model,imgs)
        partial_feature = extract_partial_feature(partial_model,global_feature,opt.part_num)
        total_partial_feature.append(partial_feature.data.cpu())
        #####
        outputs=extract_pg_global_feature(global_model,global_feature,masks)
        outputs = (outputs**2).sum(1)
            b, h, w = outputs.size()
            outputs = outputs.view(b, h * w)
            outputs = F.normalize(outputs, p=2, dim=1)
            outputs = outputs.view(b, h, w)

            
            imgs, outputs = imgs.cpu(), outputs.cpu()

            for j in range(outputs.size(0)):
                # get image name
                path = paths[j]
                imname = "./log/" + int(it)+"_"+int(j)+".jpg"

                # RGB image
                img = imgs[j, ...]
                for t, m, s in zip(img, img_mean, img_std):
                    t.mul_(s).add_(m).clamp_(0, 1)
                img_np = np.uint8(np.floor(img.numpy() * 255))
                img_np = img_np.transpose((1, 2, 0)) # (c, h, w) -> (h, w, c)

                # activation map
                am = outputs[j, ...].numpy()
                am = cv2.resize(am, (width, height))
                am = 255 * (am - np.min(am)) / (
                    np.max(am) - np.min(am) + 1e-12
                )
                am = np.uint8(np.floor(am))
                am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

                # overlapped
                overlapped = img_np*0.3 + am*0.7
                overlapped[overlapped > 255] = 255
                overlapped = overlapped.astype(np.uint8)

                # save images in a single figure (add white spacing between images)
                # from left to right: original image, activation map, overlapped image
                grid_img = 255 * np.ones(
                    (height, 3*width + 2*GRID_SPACING, 3), dtype=np.uint8
                )
                grid_img[:, :width, :] = img_np[:, :, ::-1]
                grid_img[:,
                         width + GRID_SPACING:2*width + GRID_SPACING, :] = am
                grid_img[:, 2*width + 2*GRID_SPACING:, :] = overlapped
                cv2.imwrite(osp.join(actmap_dir, imname ), grid_img)
        
        
    return pg_global_feature









############# Load  Trained model
print('-------test-----------')
model_structure = PCB(opt.train_classnum)
model = load_network('net',model_structure)
global_model_structure=ClassBlock(4096,opt.train_classnum,True,False,opt.hidden_dim)
global_model=load_network('global',global_model_structure)
global_model.classifier=nn.Sequential()
partial_model={}
for i in range(opt.part_num):
    part_model_=ClassBlock(2048,opt.train_classnum,True,False,opt.hidden_dim)
    partial_model[i]=load_network('partial'+str(i),part_model_)
    partial_model[i].classifier = nn.Sequential()
    partial_model[i].eval()
    partial_model[i]=partial_model[i].cuda()

# Change to test mode
model.eval()
global_model.eval()
model = model.cuda()
global_model=global_model.cuda()
##########Test data path
gallery_path=os.path.join(opt.test_dir,'gallery')
query_path=os.path.join(opt.test_dir,'query')
gllist=os.listdir(gallery_path)
ap = 0.0
count=0

####################
def main():
    print('extracting gallery feature...')
    #####
    # tgpl:total gallery part label; tgf:total gallery partial feature; tgf2:total gallery pose-guided global feature; tgl:total gallery label; tgc: total gallery camera id
   
    print('extracting query feature...')
    #####
    # tqpl:total query part label; tqf:total query partial feature; tqf2:total query pose-guided global feature; tql:total query label; tqc: total query camera id
    ####
    tqpl,tqf,tqf2,tql,tqc,query_imgs=feature_extractor(query_path,opt.query_heatmapdir,opt.query_posedir,
                                                            model,partial_model,global_model)
    print('query feature finished')

    print('CMC calculating...')

    count=0
    CMC=torch.IntTensor(len(gallery_imgs)).zero_()
    ap=0.0

    for qf,qf2,qpl,ql,qc in zip(tqf,tqf2,tqpl,tql,tqc):
        (ap_tmp, CMC_tmp),index = evaluate(qf,qf2,qpl,ql,qc,tgf,tgf2,tgpl,tgl,tgc) #
        if CMC_tmp[0]==-1:
            continue

    partial_feature=nn.AdaptiveAvgPool2d((part_num,1))(global_feature)
    partial_feature=torch.squeeze(partial_feature,-1)
    partial_feature=partial_feature.permute(0,2,1)
    return partial_feature
