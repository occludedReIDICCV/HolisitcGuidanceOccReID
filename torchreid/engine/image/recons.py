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

class ImageReconsEngine(Engine):

    def __init__(
            self,
            datamanager,
            model,
            optimizer,
            margin=0.3,
            weight_t=1,
            weight_x=1,
            weight_r=.1,
            scheduler=None,
            use_gpu=True,
            label_smooth=True,
            mmd_only=True,
    ):
        super(ImageReconsEngine, self).__init__(datamanager, model, optimizer, scheduler, use_gpu)

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
        self.mmd_only = mmd_only
        #self.mmd_simple = mmd_linear`

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
        losses_recons_s = AverageMeter()
        losses_recons_t = AverageMeter()
        losses_mmd_bc = AverageMeter()
        losses_mmd_wc = AverageMeter()
        losses_mmd_global = AverageMeter()
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
        weight_r=0.1
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
            outputs, features, recons = self.model(imgs)
            outputs_t, features_t, recons_t = self.model(imgs_t)
            #recons = recons.sigmoid()
            #recons_t = recons_t.sigmoid()
            loss_t = self._compute_loss(self.criterion_t, features, pids)
            loss_x = self._compute_loss(self.criterion_x, outputs, pids)
            loss_r1 = self.criterion_mse(imgs, recons)
            loss_r2 = self.criterion_mse(imgs_t, recons_t)
            
            #mm1 = torch.sum((imgs.reshape(imgs.size(0),-1) - recons.reshape(imgs.size(0),-1) ) ** 2, dim=1)
            #mm2 = torch.sum((imgs_t.reshape(imgs.size(0),-1)  - recons_t.reshape(imgs.size(0),-1) ) ** 2, dim=1)
            mm1 = recons.reshape(recons.size(0),-1)
            mm2 = recons_t.reshape(recons_t.size(0),-1)
            loss_mmd_recons = mmd_linear(mm1, mm2)
            loss = loss_t + loss_x + weight_r*loss_r1 + loss_mmd_recons + weight_r*loss_r2

            if epoch > 24:
                loss_mmd_wc, loss_mmd_bc, loss_mmd_global = self._compute_loss(self.criterion_mmd, features, features_t)
                #loss = loss_t + loss_x + loss_mmd_bc + loss_mmd_wc
                
                loss = loss_t + loss_x + loss_mmd_global + loss_mmd_bc + loss_mmd_wc + weight_r*loss_r1  + weight_r*loss_r2 +0.1*loss_mmd_recons


                if self.mmd_only:
                    loss_t = torch.tensor(0)
                    loss_x = torch.tensor(0)
                    #loss = loss_mmd_bc + loss_mmd_wc
                    loss = loss_mmd_bc + loss_mmd_wc + loss_mmd_global + weight_r*loss_r1  +  weight_r*loss_r2 + loss_mmd_recons



            loss.backward()
            self.optimizer.step()
# -------------------------------------------------------------------------------------------------------------------- #

            batch_time.update(time.time() - end)
            losses_triplet.update(loss_t.item(), pids.size(0))
            losses_softmax.update(loss_x.item(), pids.size(0))
            losses_recons_s.update(loss_r1.item(), pids.size(0))
            losses_recons_t.update(loss_mmd_recons.item(), pids.size(0))
            if epoch > 24:
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
                    'Loss_t {losses1.val:.4f} ({losses1.avg:.4f})\t'
                    'Loss_x {losses2.val:.4f} ({losses2.avg:.4f})\t'
                    'Loss_mmd_wc {losses3.val:.4f} ({losses3.avg:.4f})\t'
                    'Loss_mmd_bc {losses4.val:.4f} ({losses4.avg:.4f})\t'
                    'Loss_mmd_reconsl {losses5.val:.4f} ({losses5.avg:.4f})\t'
                    'eta {eta}'.format(
                        epoch + 1,
                        max_epoch,
                        batch_idx + 1,
                        num_batches,
                        batch_time=batch_time,
                        losses1=losses_triplet,
                        losses2=losses_softmax,
                        losses3=losses_mmd_wc,
                        losses4=losses_mmd_bc,
                        losses5=losses_recons_t,
                        eta=eta_str
                    )
                )

            if writer is not None:
                n_iter = epoch * num_batches + batch_idx
                writer.add_scalar('Train/Time', batch_time.avg, n_iter)
                writer.add_scalar('Train/Loss_triplet', losses_triplet.avg, n_iter)
                writer.add_scalar('Train/Loss_softmax', losses_softmax.avg, n_iter)
                writer.add_scalar('Train/Loss_mmd_bc', losses_mmd_bc.avg, n_iter)
                writer.add_scalar('Train/Loss_mmd_wc', losses_mmd_wc.avg, n_iter)
                writer.add_scalar('Train/Loss_mmd_global', losses_mmd_global.avg, n_iter)
                writer.add_scalar('Train/Loss_recons_s', losses_recons_s.avg, n_iter)
                writer.add_scalar('Train/Loss_recons_t', losses_recons_t.avg, n_iter)
                   
                

            end = time.time()

        if self.scheduler is not None:
            self.scheduler.step()

        print_distri = False

        if print_distri:

            instances = self.datamanager.train_loader.sampler.num_instances
            batch_size = self.datamanager.train_loader.batch_size
            feature_size = 2048 # features_t.shape[1]  # 2048
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
            plt.ylabel('Frequence of apparition')
            plt.title('Source Domain')
            plt.legend()
            plt.show()

            b_ct = [x.cpu().detach().item() for x in bct.flatten() if x > 0.1]
            w_ct = [x.cpu().detach().item() for x in wct.flatten() if x > 0.1]
            data_bc = norm.rvs(b_ct)
            sns.distplot(data_bc, bins='auto', fit=norm, kde=False, label='from the same class (within class)')
            data_wc = norm.rvs(w_ct)
            sns.distplot(data_wc, bins='auto', fit=norm, kde=False, label='from different class (between class)')
            plt.xlabel('Euclidean distance')
            plt.ylabel('Frequence of apparition')
            plt.title('Target Domain')
            plt.legend()
            plt.show()
