import torchreid
import torch.nn as nn
from config import Config
from solver import make_optimizer, WarmupMultiStepLR
import torchvision.transforms as T

import torchvision.transforms as T
source = 'market1501'
target = 'occluded-reid'

batch_size = 64


        
datamanager = torchreid.data.ImageDataManager(
        root='reid-data',
        sources=source,
        targets=target,
        height=384,
        width=128,
        batch_size_train=batch_size,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop'],
        num_instances=4,
        train_sampler='RandomIdentitySampler',
        load_train_targets=True
)
"""
datamanager2 = torchreid.data.ImageDataManager(
        root='reid-data',
        sources=source,
        targets=target,
        height=384,
        width=128,
        batch_size_train=batch_size,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop'],
        num_instances=4,
        train_sampler='RandomIdentitySampler',
        load_train_targets=True
)"""

model = torchreid.models.build_model(
        name='multigrain',
        num_classes=datamanager.num_train_pids,
        loss='triplet',
        pretrained=True
)

#model = model.cuda()
model = nn.DataParallel(model).cuda() # Comment previous line and uncomment this line for multi-gpu use
cfg = Config()
#optimizer = make_optimizer(cfg, model)

optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',
        lr=0.00035
)

scheduler = WarmupMultiStepLR(optimizer, cfg.STEPS, cfg.GAMMA,
                                  cfg.WARMUP_FACTOR,
                                  cfg.WARMUP_EPOCHS, cfg.WARMUP_METHOD)
scheduler =scheduler# torchreid.optim.build_lr_scheduler(
       # optimizer,
       # lr_scheduler='single_step',
       # stepsize=50
#) #Model 110 best for occluded dataset on partia 75 and 78.5conda
# We use pretrained model to continue the training on the target domain # "log/first_try_recons_var_r3_0/model.pth.tar-70" best model for Partial
start_epoch = torchreid.utils.resume_from_checkpoint(
        "log/first_try_recons_var_r3_0/model.pth.tar-80",# '/export/livia/home/vision/mkiran/work/Person_Reid/Video_Person/Domain_Adapt/D-MMD/log/upper_bound/source_market1501_target_market1501/model.pth.tar-30', #'log/source_training/' + source + '/model/model.pth.tar-30',
        model,
        optimizer
)
model = model.cuda()
#model = nn.DataParallel(model)  # Comment previous line and uncomment this line for multi-gpu use




engine = torchreid.engine.ImageMmdAEEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True,
        datamanager2=None,
      
)


# Define the lower bound - test without adaptation
engine.run(
        test_only=True,
)

# Start the domain adaptation
engine.run(
        save_dir='log/first_try_recons_var_r3_0',
        max_epoch=150,
        eval_freq=5,
        print_freq=10,
        test_only=False,
        visrank=False,
        start_epoch=start_epoch
)
