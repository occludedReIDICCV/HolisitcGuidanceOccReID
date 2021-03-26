import torchreid
import torch.nn as nn
source = 'market1501'
target = source

batch_size = 128

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
        load_train_targets=False
)

model = torchreid.models.build_model(
        name='multigrain',
        num_classes=datamanager.num_train_pids,
        loss='triplet',
        pretrained=True
)

#model = model.cuda()
model = nn.DataParallel(model).cuda()  # Comment previous line and uncomment this line for multi-gpu use

optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',
        lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=50
)


engine = torchreid.engine.ImageTripletAEEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True,
)

"""start_epoch = torchreid.utils.resume_from_checkpoint(
      'log/source_training_parts_clean/market1501/model.pth.tar-25',
       model,
       optimizer,
)"""

#engine.run(
#       test_only=True,
#       dist_metric='cosine'
#)


engine.run(
        save_dir='log/source_training_parts_clean/' + source, #_Duke_Maks_Based
        max_epoch=80,
        eval_freq=5,
        print_freq=10,
        test_only=False,
        visrank=False,
        start_epoch=0
)
