from __future__ import division, print_function, absolute_import
import time
import datetime

from torchreid import metrics
from torchreid.utils import (
    AverageMeter, open_all_layers, open_specified_layers
)
from torchreid.losses import MaximumMeanDiscrepancy
import torch
from functools import partial
from torch.autograd import Variable
from ..engine import Engine
from torchreid.metrics import compute_distance_matrix,compute_weight_distance_matrix_NOMASK
import numpy as np
import pickle
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from torchreid.losses import TripletLoss, CrossEntropyLoss
import random
import math
from torch import nn
import torch.nn.functional as F
from collections import deque
import torchvision.models as models
from torchreid import metrics
from torch.nn.modules import loss
import torch
from torch.nn import init
import torch.backends.cudnn as cudnn
from torch.nn import CrossEntropyLoss as CrossEntropyLossTorch
import torchvision.transforms as T
import torchvision
from .functional import to_tensor, augmentations_all

from torch import nn

### Squeeze and Excitation Class definition
class SENet(nn.Module):
    def __init__(self, channel, reduction_ratio =16):
        super(SENet, self).__init__()
        ### Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        ### Fully Connected Multi-Layer Perceptron (FC-MLP)
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.mlp(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Loss(loss._Loss):
    def __init__(self, num_classes,use_gpu,label_smooth):
        super(Loss, self).__init__()
        self.cross_entropy_loss = CrossEntropyLossTorch()
        self.triplet_loss = TripletLoss(margin=1.2)

    def forward(self, outputs, labels):
        

        Triplet_Loss = [self.triplet_loss(output, labels) for output in outputs[1:4]]
        Triplet_Loss = sum(Triplet_Loss) / len(Triplet_Loss)

        CrossEntropy_Loss = [self.cross_entropy_loss(output, labels) for output in outputs[4:]]
        CrossEntropy_Loss = sum(CrossEntropy_Loss) / len(CrossEntropy_Loss)

        loss_sum = Triplet_Loss + 2 * CrossEntropy_Loss

        print('\rtotal loss:%.2f  Triplet_Loss:%.2f  CrossEntropy_Loss:%.2f' % (
            loss_sum.data.cpu().numpy(),
            Triplet_Loss.data.cpu().numpy(),
            CrossEntropy_Loss.data.cpu().numpy()),
              end=' ')
        return loss_sum

def rand_bbox(self,img,img2, lam):
            size=img.shape
            W = size[1]
            H = size[2]
            cut_rat = np.sqrt(1. - lam)
            cut_w = np.int(W * cut_rat)
            cut_h = np.int(H * cut_rat)

            # uniform
            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)
            #rand_index = torch.randperm(imgs.size()[0])
            img[ :, bbx1:bbx2, bby1:bby2] = img2[ :, bbx1:bbx2, bby1:bby2]
            return img
class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img, img2):

        if random.uniform(0, 1) >= self.probability:
            return img,1
        
        if random.uniform(0, 1) >= self.probability:
            lam=0.6
            img = rand_bbox(self,img,img2, lam)
            return img, 0

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img,0

        return img,1

class FC_Model_Binary(nn.Module):
    def __init__(self):
        super(FC_Model_Binary, self).__init__()
        feats = 256
        student_classes=1
        self.fc_binary1 = nn.Linear(feats, student_classes)
        self.fc_binary2 = nn.Linear(feats, student_classes)
        self.fc_binary3 = nn.Linear(feats, student_classes)
        self._init_fc(self.fc_binary1)
        self._init_fc(self.fc_binary2)
        self._init_fc(self.fc_binary3)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)
    
    def forward(self, output2):
        ret1 = self.fc_binary1(output2[1])
        ret2 = self.fc_binary1(output2[2])
        ret3 = self.fc_binary1(output2[3])

        return ret1,ret2,ret3
        

class FC_Model(nn.Module):
    def __init__(self):
        super(FC_Model, self).__init__()
        feats = 256
        
        student_classes = 751#505
        self.fc_id_2048_0 = nn.Linear(feats, student_classes)
        self.fc_id_2048_1 = nn.Linear(feats, student_classes)
        self.fc_id_2048_2 = nn.Linear(feats, student_classes)

        self.fc_id_256_1_0 = nn.Linear(feats, student_classes)
        self.fc_id_256_1_1 = nn.Linear(feats, student_classes)
        self.fc_id_256_2_0 = nn.Linear(feats, student_classes)
        self.fc_id_256_2_1 = nn.Linear(feats, student_classes)
        self.fc_id_256_2_2 = nn.Linear(feats, student_classes)

        self._init_fc(self.fc_id_2048_0)
        self._init_fc(self.fc_id_2048_1)
        self._init_fc(self.fc_id_2048_2)

        self._init_fc(self.fc_id_256_1_0)
        self._init_fc(self.fc_id_256_1_1)
        self._init_fc(self.fc_id_256_2_0)
        self._init_fc(self.fc_id_256_2_1)
        self._init_fc(self.fc_id_256_2_2)

   
    
   
    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

    def forward(self, output2):
        fg_p1, fg_p2, fg_p3,f0_p2,f1_p2,f0_p3,f1_p3,f2_p3 = output2
       


        l_p1 = self.fc_id_2048_0(fg_p1)
        l_p2 = self.fc_id_2048_1(fg_p2)
        l_p3 = self.fc_id_2048_2(fg_p3)

        l0_p2 = self.fc_id_256_1_0(f0_p2)
        l1_p2 = self.fc_id_256_1_1(f1_p2)
        l0_p3 = self.fc_id_256_2_0(f0_p3)
        l1_p3 = self.fc_id_256_2_1(f1_p3)
        l2_p3 = self.fc_id_256_2_2(f2_p3)
        
    

        predict = torch.cat([fg_p1, fg_p2, fg_p3, f0_p2, f1_p2, f0_p3, f1_p3, f2_p3], dim=1)
        return predict, fg_p1, fg_p2, fg_p3, l_p1, l_p2, l_p3, l0_p2, l1_p2, l0_p3, l1_p3, l2_p3




class ImageMmdAEEngine(Engine):

    def __init__(
            self,
            datamanager,
            model,
            optimizer,
            margin=0.27,
            weight_t=1,
            weight_x=1,
            weight_r = 0.0000000001, #lambda
            scheduler=None,
            use_gpu=True,
            label_smooth=True,
            mmd_only=True,
            datamanager2=None,
    ):
        super(ImageMmdAEEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu, mmd_only,datamanager2)

        self.optimizer.zero_grad()
        self.mmd_only = mmd_only ###
        self.weight_t = weight_t
        self.weight_x = weight_x
        self.weight_r = weight_r

        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        self.criterion_mmd = MaximumMeanDiscrepancy(
            instances=self.datamanager.train_loader.sampler.num_instances,
            batch_size=self.datamanager.train_loader.batch_size,
            global_only=False,
            distance_only=True,
            all=False,
            
        )
        self.criterion_mse = torch.nn.MSELoss()
        self.random = RandomErasing(probability=0.5,sl=0.07)
        self.randomt = RandomErasing(probability=0.5,sl=0.01)
        self.mgn_loss = Loss(num_classes=self.datamanager.num_train_pids,use_gpu=self.use_gpu,label_smooth=label_smooth)
        self.mgn_targetPredict =FC_Model().cuda() 
         
        self.BCE_criterion = torch.nn.BCEWithLogitsLoss()
     
    

    def rand_bbox(self,img,img2,size, lam):
            lam = 0.8
            size=img.shape
            W = size[1]
            H = size[2]
            cut_rat = np.sqrt(1. - lam)
            cut_w = np.int(W * cut_rat)
            cut_h = np.int(H * cut_rat)

            # uniform
            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)
            #rand_index = torch.randperm(imgs.size()[0])
            img[ :, bbx1:bbx2, bby1:bby2] = img2[ :, bbx1:bbx2, bby1:bby2]
            return img

    def get_local_correl(self,local_feat):
        
        local_feat =local_feat.reshape(local_feat.size(0)*6, 2048) 
        final_dis_mat = compute_distance_matrix(local_feat,local_feat)
          
        
         
        return final_dis_mat

    

    def _construct_em_layer(self, fc_dims, input_dim, dropout_p=0.5):
        """
        Construct fully connected layer
        - fc_dims (list or tuple): dimensions of fc layers, if None,
                                   no fc layers are constructed
        - input_dim (int): input dimension
        - dropout_p (float): dropout probability, if None, dropout is unused
        """
        
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        layers = []
      
        # layers.append(nn.Linear(input_dim, fc_dims))
        layers.append(nn.Conv2d(input_dim, fc_dims, 1, stride=1, padding=0))
        layers.append(nn.BatchNorm2d(fc_dims))
        layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.Dropout(p=dropout_p))

        # self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def train(
            self,
            epoch,
            max_epoch,
            writer,
            print_freq=10,
            fixbase_epoch=0,
            open_layers=None,
    ):
        losses_triplet = AverageMeter()
        losses_softmax = AverageMeter()
        losses_mmd_bc = AverageMeter()
        losses_mmd_wc = AverageMeter()
        losses_mmd_global = AverageMeter()
        losses_recons = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        

        self.model.train()
        self.mgn_targetPredict.train()
       
        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print(
                '* Only train {} (epoch: {}/{})'.format(
                    open_layers, epoch + 1, fixbase_epoch
                )
            )
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)
            open_all_layers(self.mgn_targetPredict)
            print("All open layers!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        num_batches = len(self.train_loader)
        end = time.time()
       
# -------------------------------------------------------------------------------------------------------------------- #
        for batch_idx, (data, data_t) in enumerate(zip(self.train_loader, self.train_loader_t)):
            data_time.update(time.time() - end)
            

            imgs, pids = self._parse_data_for_train(data)
            imgs_clean =  imgs.clone().cuda()
            lam=0
            imgs_t, pids_t = self._parse_data_for_train(data_t)
            imagest_orig=imgs_t.cuda()
            labels=[]
            labelss=[]
            random_indexS = np.random.randint(0, imgs.size()[0])
            random_indexT = np.random.randint(0, imgs_t.size()[0])
            if epoch > 10 and epoch < 35:
                
                for i, img in enumerate(imgs):
                  
                   randmt = RandomErasing(probability=0.5,sl=0.07, sh=0.22)
                  
                   imgs[i],p = randmt(img, imgs[random_indexS])
                   labelss.append(p)
               
            if epoch >= 35:
                randmt = RandomErasing(probability=0.5,sl=0.1, sh=0.25)
                for i, img in enumerate(imgs):
                  
                   imgs[i],p = randmt(img,imgs[random_indexS])
                   labelss.append(p)

            





            
            if epoch > 10 and epoch < 35:
                randmt = RandomErasing(probability=0.5,sl=0.1, sh=0.2)
                for i, img in enumerate(imgs_t):
                   
                   imgs_t[i],p = randmt(img,imgs_t[random_indexT])
                   labels.append(p)
               
            if epoch >= 35 and epoch < 75:
                randmt = RandomErasing(probability=0.5,sl=0.2, sh=0.3)
                for i, img in enumerate(imgs_t):
                  
                   imgs_t[i],p = randmt(img,imgs_t[random_indexT])
                   labels.append(p)

            if epoch >= 75:
                randmt = RandomErasing(probability=0.5,sl=0.2, sh=0.35)
                for i, img in enumerate(imgs_t):
                   
                  
                   imgs_t[i],p = randmt(img,imgs_t[random_indexT])
                   labels.append(p)
           
            binary_labels = torch.tensor(np.asarray(labels)).cuda()
            binary_labelss = torch.tensor(np.asarray(labelss)).cuda()
            
               
            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()
            if self.use_gpu:
                imgs_transformed = imgs_t.cuda()

            

            self.optimizer.zero_grad()
           
            imgs_clean = imgs
            outputs, output2, recons,bcc1, bocc2,bocc3 = self.model(imgs)

            occ_losss1 = self.BCE_criterion(bcc1.squeeze(1),binary_labelss.float() )
            occ_losss2 = self.BCE_criterion(bocc2.squeeze(1),binary_labelss.float() )
            occ_losss3 = self.BCE_criterion(bocc3.squeeze(1),binary_labelss.float() )

            occ_s  = occ_losss1  +occ_losss2+occ_losss3
       
           

          

            ##############CUT MIX#################################3333
            """bbx1, bby1, bbx2, bby2 = self.rand_bbox(imgs.size(), lam)
            rand_index = torch.randperm(imgs.size()[0]).cuda()
            imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
            targeta = pids
            targetb = pids[rand_index]"""

            ##############CUT MIX#################################3333

            outputs_t, output2_t, recons_t,bocct1, bocct2,bocct3 = self.model(imagest_orig)
            outputs_t = self.mgn_targetPredict(output2_t)
           


            loss_reconst=self.criterion_mse(recons_t, imagest_orig)
            loss_recons=self.criterion_mse(recons, imgs_clean)

         
            occ_loss1 = self.BCE_criterion(bocct1.squeeze(1),binary_labels.float() )
            occ_loss2 = self.BCE_criterion(bocct2.squeeze(1),binary_labels.float() )
            occ_loss3 = self.BCE_criterion(bocct3.squeeze(1),binary_labels.float() )
            occ_t = occ_loss1 + occ_loss2 + occ_loss3
            pids_t = pids_t.cuda()
            loss_x = self.mgn_loss(outputs, pids)
            loss_x_t = self.mgn_loss(outputs_t, pids_t)
            #loss_x_t = self._compute_loss(self.criterion_x, y, targeta)  #*lam + self._compute_loss(self.criterion_x, y, targetb)*(1-lam)
            #loss_t_t = self._compute_loss(self.criterion_t, features_t, targeta)*lam + self._compute_loss(self.criterion_t, features_t, targetb)*(1-lam)
                      
         
            if epoch > 10:

                loss_mmd_wc, loss_mmd_bc, loss_mmd_global = self._compute_loss(self.criterion_mmd, outputs[0],  outputs_t[0])
                #loss_mmd_wc1, loss_mmd_bc1, loss_mmd_global1  = self._compute_loss(self.criterion_mmd, outputs[2], outputs_t[2])
                #loss_mmd_wc3, loss_mmd_bc3, loss_mmd_global3  = self._compute_loss(self.criterion_mmd, outputs[3], outputs_t[3])
                
                #loss_mmd_wcf  = loss_mmd_wc+loss_mmd_wc1+loss_mmd_wc3
                #loss_mmd_bcf  = loss_mmd_bc+loss_mmd_bc1+loss_mmd_bc3
                #loss_mmd_globalf  = loss_mmd_global+loss_mmd_global1+loss_mmd_global3
                

                
                #print(loss_mmd_bc.item())

                l_joint =  1.5*loss_x_t  +loss_x +loss_reconst+loss_recons  #self.weight_r*loss_recons+ + loss_x + loss_t 
                #loss = loss_t + loss_x + loss_mmd_bc + loss_mmd_wc
                l_d =   0.5*loss_mmd_bc + 0.8*loss_mmd_wc    +loss_mmd_global #+loss_mmd_bc1 + loss_mmd_wc1    +loss_mmd_global1 +loss_mmd_bc3 + loss_mmd_wc3   +loss_mmd_global3
                loss =  0.3*l_d + 0.7*l_joint +0.2*occ_t + 0.1*occ_s

                

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
# -------------------------------------------------------------------------------------------------------------------- #

            batch_time.update(time.time() - end)
            #losses_triplet.update(loss_t.item(), pids.size(0))
            losses_softmax.update(loss_x_t.item(), pids.size(0))
            #losses_recons.update(loss_recons.item(), pids.size(0))
            if epoch > 10:
                losses_mmd_bc.update(loss_mmd_bc.item(), pids.size(0))
                losses_mmd_wc.update(loss_mmd_wc.item(), pids.size(0))
                losses_mmd_global.update(loss_mmd_global.item(), pids.size(0))

            if (batch_idx + 1) % print_freq == 0:
                # estimate remaining time
                eta_seconds = batch_time.avg * (
                        num_batches - (batch_idx + 1) + (max_epoch -
                                                         (epoch + 1)) * num_batches
                )
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'Epoch: [{0}/{1}][{2}/{3}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    #'Loss_t {losses1.val:.4f} ({losses1.avg:.4f})\t'
                    'Loss_x {losses2.val:.4f} ({losses2.avg:.4f})\t'
                    'Loss_mmd_wc {losses3.val:.4f} ({losses3.avg:.4f})\t'
                    'Loss_mmd_bc {losses4.val:.4f} ({losses4.avg:.4f})\t'
                    'Loss_mmd_global {losses5.val:.4f} ({losses5.avg:.4f})\t'
                    #'Loss_recons {losses6.val:.4f} ({losses6.avg:.4f})\t'
                    'eta {eta}'.format(
                        epoch + 1,
                        max_epoch,
                        batch_idx + 1,
                        num_batches,
                        batch_time=batch_time,
                        #losses1=losses_triplet,
                        losses2=losses_softmax,
                        losses3=losses_mmd_wc,
                        losses4=losses_mmd_bc,
                        losses5=losses_mmd_global,
                        #losses6 = losses_recons,
                        eta=eta_str
                    )
                )
            writer = None
            if writer is not None:
                n_iter = epoch * num_batches + batch_idx
                writer.add_scalar('Train/Time', batch_time.avg, n_iter)
                writer.add_scalar('Train/Loss_triplet', losses_triplet.avg, n_iter)
                writer.add_scalar('Train/Loss_softmax', losses_softmax.avg, n_iter)
                writer.add_scalar('Train/Loss_mmd_bc', losses_mmd_bc.avg, n_iter)
                writer.add_scalar('Train/Loss_mmd_wc', losses_mmd_wc.avg, n_iter)
                writer.add_scalar('Train/Loss_mmd_global', losses_mmd_global.avg, n_iter)
                writer.add_scalar(
                    'Train/Lr', self.optimizer.param_groups[0]['lr'], n_iter
                )

            end = time.time()

        if self.scheduler is not None:
            self.scheduler.step()
        print_distri = True

        if print_distri:

            instances = self.datamanager.test_loader.query_loader.num_instances
            batch_size = self.datamanager.test_loader.batch_size
            feature_size = outputs[0].size(1) # features_t.shape[1]  # 2048
            features_t = outputs_t[0]
            features = outputs[0]
            t = torch.reshape(features_t, (int(batch_size / instances), instances, feature_size))
 
            #  and compute bc/wc euclidean distance
            bct = compute_distance_matrix(t[0], t[0])
            wct = compute_distance_matrix(t[0], t[1])
            for i in t[1:]:
                bct = torch.cat((bct, compute_distance_matrix(i, i)))
                for j in t:
                    if j is not i:
                        wct = torch.cat((wct, compute_distance_matrix(i, j)))

            s = torch.reshape(features, (int(batch_size / instances), instances, feature_size))
            bcs = compute_distance_matrix(s[0], s[0])
            wcs = compute_distance_matrix(s[0], s[1])
            for i in s[1:]:
                bcs = torch.cat((bcs, compute_distance_matrix(i, i)))
                for j in s:
                    if j is not i:
                        wcs = torch.cat((wcs, compute_distance_matrix(i, j)))

            bcs = bcs.detach()
            wcs = wcs.detach()

            b_c = [x.cpu().detach().item() for x in bcs.flatten() if x > 0.000001]
            w_c = [x.cpu().detach().item() for x in wcs.flatten() if x > 0.000001]
            data_bc = norm.rvs(b_c)
            sns.distplot(data_bc, bins='auto', fit=norm, kde=False, label='from the same class (within class)')
            data_wc = norm.rvs(w_c)
            sns.distplot(data_wc, bins='auto', fit=norm, kde=False, label='from different class (between class)')
            plt.xlabel('Euclidean distance')
            plt.ylabel('Frequency')
            plt.title('Source Domain')
            plt.legend()
            plt.savefig("Source.png")
            plt.clf()
            b_ct = [x.cpu().detach().item() for x in bct.flatten() if x > 0.1]
            w_ct = [x.cpu().detach().item() for x in wct.flatten() if x > 0.1]
            data_bc = norm.rvs(b_ct)
            sns.distplot(data_bc, bins='auto', fit=norm, kde=False, label='from the same class (within class)')
            data_wc = norm.rvs(w_ct)
            sns.distplot(data_wc, bins='auto', fit=norm, kde=False, label='from different class (between class)')
            plt.xlabel('Euclidean distance')
            plt.ylabel('Frequency')
            plt.title('Target Domain')
            plt.legend()
            plt.savefig("Target.png")
