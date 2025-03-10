import torch
from torchvision import transforms
from functools import partial
from easydict import EasyDict as edict
from albumentations import Resize, HorizontalFlip
from kornia.color import RgbToHsv, HsvToRgb
from torch.nn import init

from iharm.data.compose import ComposeDatasetUpsample
from iharm.data.hdataset import HDatasetUpsample
from iharm.data.transforms import HCompose, RandomCropNoResize

from iharm.model.base.aict_net import AICT
from iharm.model.losses import MaskWeightedMSE, SCS_CR_loss, ColorDistance, CoordDistance
from iharm.model.metrics import DenormalizedPSNRMetric_FR, DenormalizedMSEMetric_FR
from iharm.engine.AICT_trainer import Trainer
from iharm.mconfigs import BMCONFIGS
from iharm.utils.log import logger

def main(cfg):
    model, model_cfg, ccfg = init_model()
    train(model, cfg, model_cfg, ccfg)


def init_func(m, init_gain=0.02):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

def init_model():
    model_cfg = edict()
    model_cfg.input_normalization = {
        'mean': [0, 0, 0],
        'std': [1, 1, 1]
    }

    ccfg = BMCONFIGS['ViT_aict']
    ccfg['params']['input_normalization'] = model_cfg.input_normalization   
    model = AICT(**ccfg['params'])

    model.apply(init_func)                                                  # apply the initialization function <init_func>

    input_transform = [transforms.ToTensor()]
    if ccfg['data']['color_space'] == 'HSV':
        input_transform.append(RgbToHsv())
        input_transform.append(transforms.Normalize([0, 0, 0], [6.283, 1, 1]))
    input_transform.append(transforms.Normalize(model_cfg.input_normalization['mean'], model_cfg.input_normalization['std']))
    model_cfg.input_transform = transforms.Compose(input_transform)

    return model, model_cfg, ccfg

def train(model, cfg, model_cfg, ccfg):
    cfg.batch_size = 16 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size

    cfg.input_normalization = model_cfg.input_normalization

    loss_cfg = edict()
    loss_cfg.pixel_loss = MaskWeightedMSE(min_area=1000, pred_name='images_fullres',
                                          gt_image_name='target_images_fullres', gt_mask_name='masks_fullres')
    loss_cfg.pixel_loss_weight = 1.0

    loss_cfg.low_loss = MaskWeightedMSE(min_area=100, pred_name='images',
                                        gt_image_name='target_images', gt_mask_name='masks')
    loss_cfg.low_loss_weight = 0.01

    num_epochs = 100
    low_res_size = (256, 256)

    train_augmentator_1 = HCompose([
        RandomCropNoResize(ratio=0.5),
        HorizontalFlip(),
    ])
    train_augmentator_2 = HCompose([
        Resize(*low_res_size)
    ])

    val_augmentator_1 = None
    val_augmentator_2 = HCompose([
        Resize(*low_res_size)
    ])

    blur = False
    trainset = ComposeDatasetUpsample(
        [
            HDatasetUpsample(cfg.HFLICKR_PATH, split='train', blur_target=blur),
            HDatasetUpsample(cfg.HDAY2NIGHT_PATH, split='train', blur_target=blur),
            HDatasetUpsample(cfg.HCOCO_PATH, split='train', blur_target=blur),
            HDatasetUpsample(cfg.HADOBE5K_2048_PATH, split='train', blur_target=blur),
            #HDatasetUpsample(cfg.CC_PATH, split='train', blur_target=blur),
        ],
        augmentator_1=train_augmentator_1,
        augmentator_2=train_augmentator_2,
        input_transform=model_cfg.input_transform,
        keep_background_prob=0.05,
        use_hr=True
    )

    valset = ComposeDatasetUpsample(
        [
            HDatasetUpsample(cfg.HFLICKR_PATH, split='test', blur_target=blur, mini_val=False),
            HDatasetUpsample(cfg.HDAY2NIGHT_PATH, split='test', blur_target=blur, mini_val=False),
            HDatasetUpsample(cfg.HCOCO_PATH, split='test', blur_target=blur, mini_val=False),
            HDatasetUpsample(cfg.HADOBE5K_2048_PATH, split='test', blur_target=blur, mini_val=False),
            #HDatasetUpsample(cfg.CC_PATH, split='test', blur_target=blur, mini_val=False)
        ],
        augmentator_1=val_augmentator_1,
        augmentator_2=val_augmentator_2,
        input_transform=model_cfg.input_transform,
        keep_background_prob=-1,
        use_hr=True
    )

    optimizer_params = {
        'lr': cfg.lr,
        'betas': (0.9, 0.999), 'eps': 1e-8
    }

    if cfg.local_rank == 0:
        print(optimizer_params)

    scheduler1 = partial(torch.optim.lr_scheduler.ConstantLR, factor=1)
    scheduler2 = partial(torch.optim.lr_scheduler.LinearLR, start_factor=1, end_factor=0, total_iters=50)
    lr_scheduler = lambda optimizer: torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1(optimizer=optimizer), scheduler2(optimizer=optimizer)], milestones=[50])

    if ccfg['data']['color_space'] == 'HSV':
        color_transform = transforms.Compose([HsvToRgb(), transforms.Normalize([0, 0, 0], [-6.283, 1, 1])])
    else:
        color_transform = None

    def collate_fn_FR(batch):
        # Initializae dictionary
        keys = ['images', 'masks', 'target_images', 'images_fullres', 'masks_fullres', 'target_images_fullres',
                'image_info']
        bdict = {}
        for k in keys:
            bdict[k] = []

        # Create batched dictionary
        for elem in batch:
            for key in keys:
                if key in ['masks', 'masks_fullres']:
                    elem[key] = torch.tensor(elem[key])
                bdict[key].append(elem[key])

        bdict['images'] = torch.stack(bdict['images'])
        bdict['target_images'] = torch.stack(bdict['target_images'])
        bdict['masks'] = torch.stack(bdict['masks'])

        return bdict

    trainer = Trainer(
        model, cfg, model_cfg, loss_cfg,
        trainset, valset,
        optimizer='adam',
        optimizer_params=optimizer_params,
        lr_scheduler=lr_scheduler,
        metrics=[
            DenormalizedPSNRMetric_FR(
                'images_fullres', 'target_images_fullres',
                mean=torch.tensor(cfg.input_normalization['mean'], dtype=torch.float32).view(1, 3, 1, 1),
                std=torch.tensor(cfg.input_normalization['std'], dtype=torch.float32).view(1, 3, 1, 1),
                color_transform=color_transform,
            ),
            DenormalizedMSEMetric_FR(
                'images_fullres', 'target_images_fullres',
                mean=torch.tensor(cfg.input_normalization['mean'], dtype=torch.float32).view(1, 3, 1, 1),
                std=torch.tensor(cfg.input_normalization['std'], dtype=torch.float32).view(1, 3, 1, 1),
                color_transform=color_transform,
            ),
        ],
        checkpoint_interval=50,
        image_dump_interval=0,
        color_space=ccfg['data']['color_space'],
        collate_fn=collate_fn_FR,
        random_swap=0,
        random_augment=True
    )

    start_epoch = trainer.cfg.start_epoch
    if cfg.local_rank == 0:
        logger.info(f'Starting Epoch: {start_epoch}')
        logger.info(f'Total Epochs: {num_epochs}')
    for epoch in range(start_epoch, num_epochs):
        trainer.training(epoch)
        if epoch > 70 or epoch % 10 == 1:
            trainer.validation(epoch)
