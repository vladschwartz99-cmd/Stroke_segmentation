import torch
from monai.networks.nets import UNet, AttentionUnet, BasicUNetPlusPlus, SwinUNETR
from monai.losses import DiceFocalLoss


# Устройство для вычислений
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_unet(in_channels, weights_dir=None):
    """Функция инициализации модели 3D U-Net, функции потерь и оптимизатора"""

    # Инициализации 3D U-Net
    unet = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2)
    ).to(device)

    # Загрузка весов в модель
    if weights_dir:

        unet.load_state_dict(torch.load(f'./weights/{weights_dir}/best_UNet.pth'))
        return unet

    else:

        # Инициализации функции потерь, оптимизатора и планировщика
        criterion = DiceFocalLoss(sigmoid=True, gamma=2.0, lambda_dice=1.0, lambda_focal=1.0)
        optimizer = torch.optim.AdamW(unet.parameters(), lr=0.001, weight_decay=0.00001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
        return unet, criterion, optimizer, scheduler



def get_attention_unet(in_channels, weights_dir=None):
    """Функция инициализации модели 3D Attention U-Net, функции потерь и оптимизатора"""

    # Инициализации 3D Attention U-Net
    attention_unet = AttentionUnet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2)
    ).to(device)

    # Загрузка весов в модель
    if weights_dir:

        attention_unet.load_state_dict(torch.load(f'./weights/{weights_dir}/best_AttentionUnet.pth'))
        return attention_unet

    else:

        # Инициализации функции потерь, оптимизатора и планировщика
        criterion = DiceFocalLoss(sigmoid=True, gamma=2.0, lambda_dice=1.0, lambda_focal=1.0)
        optimizer = torch.optim.AdamW(attention_unet.parameters(), lr=0.001, weight_decay=0.00001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        return attention_unet, criterion, optimizer, scheduler



def get_unet_plus_plus(in_channels, weights_dir=None):
    """Функция инициализации модели 3D U-Net ++, функции потерь и оптимизатора"""

    # Инициализации 3D U-Net ++
    unet_plus_plus = BasicUNetPlusPlus(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=1,
        features=(16, 32, 64, 128, 256, 16)
    ).to(device)

    # Загрузка весов в модель
    if weights_dir:

        unet_plus_plus.load_state_dict(torch.load(f'./weights/{weights_dir}/best_BasicUNetPlusPlus.pth'))
        return unet_plus_plus

    else:

        # Инициализации функции потерь, оптимизатора и планировщика
        criterion = DiceFocalLoss(sigmoid=True, gamma=2.0, lambda_dice=1.0, lambda_focal=1.0)
        optimizer = torch.optim.AdamW(unet_plus_plus.parameters(), lr=0.001, weight_decay=0.00001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        return unet_plus_plus, criterion, optimizer, scheduler



def get_swin_unetr(in_channels, weights_dir=None, pretrained=True):
    """Функция инициализации модели Swin UNETR, функции потерь и оптимизатора"""

    # Инициализация Swin UNETR
    swin_unetr = SwinUNETR(
        in_channels=in_channels,
        out_channels=1,
        feature_size=48,
        dropout_path_rate=0.1,
        use_checkpoint=True
    ).to(device)

    if pretrained:

        weights = torch.load(
            './weights/swin_unetr_pretrained/ssl_pretrained_weights.pth',
            map_location=device,
            weights_only=False,
        )['model']

        model_dict = swin_unetr.state_dict()

        filtered_weights = {}

        for k, v in weights.items():

            # пропускаем первый слой из-за каналов
            if 'patch_embed.proj.weight' in k:
                continue

            if k in model_dict and v.shape == model_dict[k].shape:
                filtered_weights[k] = v

        swin_unetr.load_state_dict(
            filtered_weights,
            strict=False
        )

    # Загрузка весов в модель
    if weights_dir:

        swin_unetr.load_state_dict(torch.load(f'./weights/{weights_dir}/best_SwinUNETR.pth'))
        return swin_unetr

    else:

        # Инициализации функции потерь, оптимизатора и планировщика
        criterion = DiceFocalLoss(sigmoid=True, gamma=2.0, lambda_dice=1.0, lambda_focal=1.0)
        optimizer = torch.optim.AdamW(swin_unetr.parameters(), lr=0.001, weight_decay=0.00001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        return swin_unetr, criterion, optimizer, scheduler



def get_unet_plus_plus_for_refinement(weights_dir=None):
    """Функция инициализации модели 3D U-Net ++, для подбора гиперпараметров"""

    # Инициализации 3D U-Net ++
    unet_plus_plus = BasicUNetPlusPlus(
        spatial_dims=3,
        in_channels=3,
        out_channels=1,
        features=(16, 32, 64, 128, 256, 16)
    ).to(device)

    # Загрузка весов в модель
    if weights_dir:

        unet_plus_plus.load_state_dict(torch.load(f'./weights/{weights_dir}/best_BasicUNetPlusPlus.pth'))
        return unet_plus_plus

    # Возвращение модели без весов
    else:

        return unet_plus_plus
