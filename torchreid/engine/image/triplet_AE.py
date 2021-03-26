from __future__ import division, print_function, absolute_import
import time
import datetime

from torchreid import metrics
from torchreid.utils import (
    AverageMeter, open_all_layers, open_specified_layers
)
from torchreid.losses import TripletLoss, CrossEntropyLoss
import torch
from ..engine import Engine
import random
import math
from torchreid import metrics
from torch.nn.modules import loss
from torch.nn import CrossEntropyLoss as CrossEntropyLossTorch
import numpy as np
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
            return img,1

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


class ImageTripletAEEngine(Engine):
    r"""Triplet-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        margin (float, optional): margin for triplet loss. Default is 0.3.
        weight_t (float, optional): weight for triplet loss. Default is 1.
        weight_x (float, optional): weight for softmax loss. Default is 1.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::

        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32,
            num_instances=4,
            train_sampler='RandomIdentitySampler' # this is important
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='triplet'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageTripletEngine(
            datamanager, model, optimizer, margin=0.3,
            weight_t=0.7, weight_x=1, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-triplet-market1501',
            print_freq=10
        )
    """

    def __init__(
            self,
            datamanager,
            model,
            optimizer,
            margin=0.3,
            weight_t=1,
            weight_x=1,
            weight_r = 0.0001,
            scheduler=None,
            use_gpu=True,
            label_smooth=True
    ):
        super(ImageTripletAEEngine, self
              ).__init__(datamanager, model, optimizer, scheduler, use_gpu)

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
        self.random = RandomErasing(probability=0.5)
        self.mgn_loss = Loss(num_classes=self.datamanager.num_train_pids,use_gpu=self.use_gpu,label_smooth=label_smooth)
        self.BCE_criterion = torch.nn.BCEWithLogitsLoss()

 
  
    def train(
            self,
            epoch,
            max_epoch,
            writer,
            print_freq=10,
            fixbase_epoch=0,
            open_layers=None
    ):
        losses_t = AverageMeter()
        losses_x = AverageMeter()
        losses_recons = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.model.train()
       
        open_all_layers(self.model)

        num_batches = len(self.train_loader)
        end = time.time()
        for batch_idx, data in enumerate(self.train_loader):
            data_time.update(time.time() - end)

            imgs, pids = self._parse_data_for_train(data)
            imgs_clean=imgs.clone()
            if self.use_gpu:
                imgs = imgs.cuda()
                imgs_clean = imgs_clean.cuda()
                pids = pids.cuda()
            labelss=[]
            if epoch >= 0 and epoch < 15:
                randmt = RandomErasing(probability=0.5,sl=0.07, sh=0.3)
                for i, img in enumerate(imgs):
                   
                   imgs[i],p = randmt(img)
                   labelss.append(p)
               
            if epoch >= 15:
                randmt = RandomErasing(probability=0.5,sl=0.1, sh=0.3)
                for i, img in enumerate(imgs):
                   
                   imgs[i],p = randmt(img)
                   labelss.append(p)

            binary_labels = torch.tensor(np.asarray(labelss)).cuda()
            self.optimizer.zero_grad()
            
            outputs, outputs2, recons,bin_out1,bin_out2, bin_out3 = self.model(imgs )
            loss_mse = self.criterion_mse(recons, imgs_clean)
            loss = self.mgn_loss(outputs, pids)
            
            occ_loss1 = self.BCE_criterion(bin_out1.squeeze(1),binary_labels.float() )
            occ_loss2 = self.BCE_criterion(bin_out2.squeeze(1),binary_labels.float() )
            occ_loss3 = self.BCE_criterion(bin_out3.squeeze(1),binary_labels.float() )


            loss = loss + .05*loss_mse + 0.1*occ_loss1 + 0.1*occ_loss2+0.1*occ_loss3
            #loss = self.weight_t * loss_t + self.weight_x * loss_x #+ #self.weight_r*loss_mse
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)

            #losses_t.update(loss_t.item(), pids.size(0))
            losses_x.update(loss.item(), pids.size(0))
            losses_recons.update(occ_loss1.item(), binary_labels.size(0))
            accs.update(metrics.accuracy(outputs, pids)[0].item())

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
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    #'Loss_t {loss_t.val:.4f} ({loss_t.avg:.4f})\t'
                    'Loss_x {loss_x.val:.4f} ({loss_x.avg:.4f})\t'
                    'Loss_Occlusion {loss_r.val:.4f} ({loss_r.avg:.4f})\t'             
                    'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                    'Lr {lr:.6f}\t'
                    'eta {eta}'.format(
                        epoch + 1,
                        max_epoch,
                        batch_idx + 1,
                        num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        #loss_t=losses_t,
                        loss_x=losses_x,
                        loss_r = losses_recons,
                        acc=accs,
                        lr=self.optimizer.param_groups[0]['lr'],
                        eta=eta_str
                    )
                )
            writer= None
            if writer is not None:
                n_iter = epoch * num_batches + batch_idx
                writer.add_scalar('Train/Time', batch_time.avg, n_iter)
                writer.add_scalar('Train/Data', data_time.avg, n_iter)
                writer.add_scalar('Train/Loss_t', losses_t.avg, n_iter)
                writer.add_scalar('Train/Loss_x', losses_x.avg, n_iter)
                writer.add_scalar('Train/Acc', accs.avg, n_iter)
                writer.add_scalar(
                    'Train/Lr', self.optimizer.param_groups[0]['lr'], n_iter
                )

            end = time.time()

        if self.scheduler is not None:
            self.scheduler.step()
