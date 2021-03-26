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
from torchreid.metrics import compute_distance_matrix
import numpy as np
import pickle
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from torchreid.losses import TripletLoss, CrossEntropyLoss
from .mmd_loss import MMD_loss
import random
import math
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

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

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
                return img

        return img


def pairwise_distance( x, y):

      if not len(x.shape) == len(y.shape) == 2:
          raise ValueError('Both inputs should be matrices.')

      if x.shape[1] != y.shape[1]:
          raise ValueError('The number of features should be the same.')

      x = x.view(x.shape[0], x.shape[1], 1)
      y = torch.transpose(y, 0, 1)
      output = torch.sum((x - y) ** 2, 1)
      output = torch.transpose(output, 0, 1)
      return output









def gaussian_kernel_matrix( x, y, sigmas):
        sigmas = sigmas.view(sigmas.shape[0], 1)
        beta = 1. / (2. * sigmas)
        dist = pairwise_distance(x, y).contiguous()
        dist_ = dist.view(1, -1)
        s = torch.matmul(beta, dist_.cuda())
        return torch.sum(torch.exp(-s), 0).view_as(dist)

sigmas = [
            1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
            1e3, 1e4, 1e5, 1e6
        ]
gaussian_kernel = partial(
                gaussian_kernel_matrix, sigmas=Variable(torch.cuda.FloatTensor(sigmas))
            )

def mmd_linear( f_of_X, f_of_Y):
        delta = f_of_X - f_of_Y
        loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
        return loss


def maximum_mean_discrepancy( x, y, kernel=gaussian_kernel_matrix):
      cost = torch.mean(kernel(x, x))
      cost += torch.mean(kernel(y, y))
      cost -= 2 * torch.mean(kernel(x, y))
      return cost

class ImageReconsVarEngine(Engine):

    def __init__(
            self,
            datamanager,
            model,
            optimizer,
            margin=0.3,
            weight_t=1,
            weight_x=1,
            weight_r=0.0001,
            scheduler=None,
            use_gpu=True,
            label_smooth=True,
            only_recons=True,
    ):
        super(ImageReconsVarEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu)

        self.optimizer.zero_grad()
       
        self.weight_t = weight_t
        self.weight_x = weight_x
        self.weight_r = weight_r

        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_mmd = MaximumMeanDiscrepancy(
            instances=self.datamanager.train_loader.sampler.num_instances,
            batch_size=self.datamanager.train_loader.batch_size,
            global_only=False,
            distance_only=True,
            all=False
        )
        self.only_recons = only_recons
        self.random = RandomErasing(probability=0.5) 
        self.random2 = RandomErasing(probability=0.65,sl=0.15)
        #self.mmd_simple = mmd_linear`
    
    def loss_vae(self, x, recons_x, mu, logvar):

        mse = self.criterion_mse(recons_x, x)

        # https://arxiv.org/abs/1312.6114 (Appendix B)
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)

        return mse + KLD

    def get_local_correl(self,local_feat):
        final_dis_mat = torch.zeros((local_feat.size(0), local_feat.size(1)*local_feat.size(1)))
        for i in range(local_feat.size(0)):
            temp = local_feat[i]
            dist_matrix = compute_distance_matrix(temp,temp)
            dist_matrix = dist_matrix.reshape(-1)
            final_dis_mat[i] = dist_matrix
         
        return final_dis_mat.cuda()


    def train(
            self,
            epoch,
            max_epoch,
            writer,
            print_freq=1,
            fixbase_epoch=0,
            open_layers=None,
    ):
        losses_triplet = AverageMeter()
        losses_softmax = AverageMeter()
        losses_recons_s = AverageMeter()
        losses_recons_t = AverageMeter()
        losses_mmd_bc = AverageMeter()
        losses_mmd_wc = AverageMeter()
        losses_mmd_global = AverageMeter()
        losses_local = AverageMeter()
           
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.model.train()
        if (epoch + 1) <= fixbase_epoch and open_layers is not None:
            print(
                '* Only train {} (epoch: {}/{})'.format(
                    open_layers, epoch + 1, fixbase_epoch
                )
            )
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)

        num_batches = len(self.train_loader)
        end = time.time()
        weight_r=self.weight_r
# -------------------------------------------------------------------------------------------------------------------- #
        for batch_idx, (data, data_t) in enumerate(zip(self.train_loader, self.train_loader_t)):
            data_time.update(time.time() - end)

            imgs, pids = self._parse_data_for_train(data)
            if self.use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()

            imgs_t, pids_t = self._parse_data_for_train(data_t)
            if self.use_gpu:
                imgs_t = imgs_t.cuda()

            self.optimizer.zero_grad()
            noisy_imgs = self.random(imgs)
            outputs,part_outs, features, recons,z, mean,var,local_feat = self.model(noisy_imgs)
            parts_loss = 0
            
            for i in range(len(part_outs)):
               out = part_outs[i]
               
               parts_loss+= self._compute_loss(self.criterion_x, out, pids)#  self.criterion( out, pids)
          
            parts_loss = parts_loss/len(part_outs)
            #print("local feats")
            #print(local_feat.shape)
            #print("global feats ")
            #print(local_feat.reshape(local_feat.size(0),-1).t().shape)
         
           
                
                
                
            imgs_t = self.random2(imgs_t)
            outputs_t,parts_out_t, features_t, recons_t,z_t,mean_t, var_t,local_feat_t = self.model(imgs_t)
            
            loss_t = self._compute_loss(self.criterion_t, features, pids)
            loss_x = self._compute_loss(self.criterion_x, outputs, pids)
            loss_r1 =  self.loss_vae(imgs, recons, mean,var)
            loss_r2 =  self.loss_vae(imgs_t, recons_t, mean_t,var_t)
           
            
            dist_mat_s = self.get_local_correl(local_feat)
            dist_mat_t = self.get_local_correl(local_feat_t)
           
            dist_mat_s=dist_mat_s.detach()
            local_loss = self.criterion_mmd.mmd_rbf_noaccelerate(dist_mat_s, dist_mat_t)
          
            kl_loss = torch.tensor(0)
            #loss = loss_t + loss_x + weight_r*loss_r1 +  (weight_r*2)*loss_r2 + loss_mmd_global #+ 0.1*kl_loss
            loss_mmd_wc, loss_mmd_bc, loss_mmd_global = self._compute_loss(self.criterion_mmd, features, features_t)
            loss = loss_t + loss_x  + weight_r*loss_r1 +0*loss_r2 +  loss_mmd_wc + loss_mmd_bc  + loss_mmd_global  +parts_loss#weight_r2 =0 is best
            if epoch > 10:
                
               
               
                #loss = loss_t + loss_x  + weight_r*loss_r1  + (weight_r)*loss_r2  +  loss_mmd_wc + loss_mmd_bc  + loss_mmd_global 

                if False:
                    loss_mmd_bc = torch.tensor(0)
                    loss_mmd_global = torch.tensor(0)
                    loss_mmd_wc = torch.tensor(0)
                    kl_loss = torch.tensor(0)
                    
                    #loss = loss_mmd_bc + loss_mmd_wc
                    loss = loss_t + loss_x + weight_r*loss_r1  +  (weight_r)*loss_r2  +  loss_mmd_wc + loss_mmd_bc  + loss_mmd_global



            loss.backward()
            self.optimizer.step()
# -------------------------------------------------------------------------------------------------------------------- #

            batch_time.update(time.time() - end)
            losses_triplet.update(loss_t.item(), pids.size(0))
            losses_softmax.update(loss_x.item(), pids.size(0))
            losses_recons_s.update(loss_r1.item(), pids.size(0))
            losses_recons_t.update(loss_r2.item(), pids.size(0))
            
            losses_local.update(local_loss.item(), pids.size(0))
            

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
                    'Loss_t {losses1.val:.4f} ({losses1.avg:.4f})\t'
                    'Loss_x {losses2.val:.4f} ({losses2.avg:.4f})\t'
                    'Loss_reconsS {losses4.val:.4f} ({losses4.avg:.4f})\t'
                    'Loss_reconsT {losses5.val:.4f} ({losses5.avg:.4f})\t'
                    'Loss_local {losses6.val:.4f} ({losses6.avg:.4f})\t'
                    
                    'eta {eta}'.format(
                        epoch + 1,
                        max_epoch,
                        batch_idx + 1,
                        num_batches,
                        batch_time=batch_time,
                        losses1=losses_triplet,
                        losses2=losses_softmax,
                        losses4=losses_recons_s,
                        losses5=losses_recons_t,
                        losses6=losses_local,
                        eta=eta_str
                    )
                )

            if writer is not None:
                n_iter = epoch * num_batches + batch_idx
                writer.add_scalar('Train/Time', batch_time.avg, n_iter)
                writer.add_scalar('Train/Loss_triplet', losses_triplet.avg, n_iter)
                writer.add_scalar('Train/Loss_softmax', losses_softmax.avg, n_iter)
            
                writer.add_scalar('Train/Loss_recons_s', losses_recons_s.avg, n_iter)
                writer.add_scalar('Train/Loss_recons_t', losses_recons_t.avg, n_iter)
                   
                

            end = time.time()

        if self.scheduler is not None:
            self.scheduler.step()

        print_distri = False
       
        if print_distri:
            print("Printing distribution")
            instances = self.datamanager.train_loader.sampler.num_instances
            batch_size = self.datamanager.train_loader.batch_size
            feature_size = 1024 # features_t.shape[1]  # 2048
            #print("local feature size!!!")
            #print(local_feat_t.shape)
            local_feat_t = local_feat_t.reshape(local_feat_t.size(0), -1)
            t = torch.reshape(local_feat_t, (int(batch_size / instances), instances, feature_size))

            #  and compute bc/wc euclidean distance
            bct = compute_distance_matrix(t[0], t[0])
            wct = compute_distance_matrix(t[0], t[1])
            for i in t[1:]:
                bct = torch.cat((bct, compute_distance_matrix(i, i)))
                for j in t:
                    if j is not i:
                        wct = torch.cat((wct, compute_distance_matrix(i, j)))

            s = torch.reshape(local_feat, (int(batch_size / instances), instances, feature_size))
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
            plt.ylabel('Frequence of Occurance')
            plt.title('Source Domain')
            plt.legend()
            plt.savefig("/export/livia/home/vision/mkiran/work/Person_Reid/Video_Person/Domain_Adapt/D-MMD/figs/Non_Occluded_distribution.png")
            plt.clf()

            b_ct = [x.cpu().detach().item() for x in bct.flatten() if x > 0.1]
            w_ct = [x.cpu().detach().item() for x in wct.flatten() if x > 0.1]
            data_bc = norm.rvs(b_ct)
            sns.distplot(data_bc, bins='auto', fit=norm, kde=False, label='from the same class (within class)')
            data_wc = norm.rvs(w_ct)
            sns.distplot(data_wc, bins='auto', fit=norm, kde=False, label='from different class (between class)')
            plt.xlabel('Euclidean distance')
            plt.ylabel('Frequence of apparition')
            plt.title('Non-Occluded Data Domain')
            plt.legend()
            plt.savefig("/export/livia/home/vision/mkiran/work/Person_Reid/Video_Person/Domain_Adapt/D-MMD/figs/Occluded_distribution.png")
            plt.clf()
