import torch
from monai.networks.nets import UNet, AttentionUnet, BasicUNetPlusPlus, SwinUNETR
from monai.losses import DiceFocalLoss


# Устройство для вычислений
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_unet(in_channels, load_weights=False):
    """Функция инициализации модели 3D U-Net, функции потерь и оптимизатора"""

    # Инициализации 3D U-Net
    unet = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2)
    ).to(device)

    if load_weights:

        # Загрузка весов в модель
        if in_channels == 2:

            unet.load_state_dict(torch.load('./weights/dwi_adc_models/best_UNet.pth'))
            return unet

        else:

            unet.load_state_dict(torch.load('./weights/dwi_adc_flair_models/best_UNet.pth'))
            return unet

    else:

        # Инициализации функции потерь, оптимизатора и планировщика
        criterion = DiceFocalLoss(sigmoid=True, gamma=2.0, lambda_dice=1.0, lambda_focal=1.0)
        optimizer = torch.optim.AdamW(unet.parameters(), lr=0.001, weight_decay=0.00001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
        return unet, criterion, optimizer, scheduler



def get_attention_unet(in_channels, load_weights=False):
    """Функция инициализации модели 3D Attention U-Net, функции потерь и оптимизатора"""

    # Инициализации 3D Attention U-Net
    attention_unet = AttentionUnet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2)
    ).to(device)

    if load_weights:

        # Загрузка весов в модель
        if in_channels == 2:

            attention_unet.load_state_dict(torch.load('./weights/dwi_adc_models/best_AttentionUnet.pth'))
            return attention_unet

        else:

            attention_unet.load_state_dict(torch.load('./weights/dwi_adc_flair_models/best_AttentionUnet.pth'))
            return attention_unet

    else:

        # Инициализации функции потерь, оптимизатора и планировщика
        criterion = DiceFocalLoss(sigmoid=True, gamma=2.0, lambda_dice=1.0, lambda_focal=1.0)
        optimizer = torch.optim.AdamW(attention_unet.parameters(), lr=0.001, weight_decay=0.00001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        return attention_unet, criterion, optimizer, scheduler



def get_unet_plus_plus(in_channels, load_weights=False):
    """Функция инициализации модели 3D U-Net ++, функции потерь и оптимизатора"""

    # Инициализации 3D U-Net ++
    unet_plus_plus = BasicUNetPlusPlus(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=1,
        features=(16, 32, 64, 128, 256, 16)
    ).to(device)

    if load_weights:

        # Загрузка весов в модель
        if in_channels == 2:

            unet_plus_plus.load_state_dict(torch.load('./weights/dwi_adc_models/best_BasicUNetPlusPlus.pth'))
            return unet_plus_plus

        else:

            unet_plus_plus.load_state_dict(torch.load('./weights/dwi_adc_flair_models/best_BasicUNetPlusPlus.pth'))
            return unet_plus_plus

    else:

        # Инициализации функции потерь, оптимизатора и планировщика
        criterion = DiceFocalLoss(sigmoid=True, gamma=2.0, lambda_dice=1.0, lambda_focal=1.0)
        optimizer = torch.optim.AdamW(unet_plus_plus.parameters(), lr=0.001, weight_decay=0.00001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        return unet_plus_plus, criterion, optimizer, scheduler



def get_swin_unetr(in_channels, load_weights=False):
    """Функция инициализации модели Swin UNETR, функции потерь и оптимизатора"""

    # Инициализация Swin UNETR
    swin_unetr = SwinUNETR(
        in_channels=in_channels,
        out_channels=1,
        feature_size=24,
        dropout_path_rate=0.1,
        use_checkpoint=True
    ).to(device)

    if load_weights:

        # Загрузка весов в модель
        if in_channels == 2:

            swin_unetr.load_state_dict(torch.load('./weights/dwi_adc_models/best_SwinUNETR.pth'))
            return swin_unetr

        else:

            swin_unetr.load_state_dict(torch.load('./weights/dwi_adc_flair_models/best_SwinUNETR.pth'))
            return swin_unetr

    else:

        # Инициализации функции потерь, оптимизатора и планировщика
        criterion = DiceFocalLoss(sigmoid=True, gamma=2.0, lambda_dice=1.0, lambda_focal=1.0)
        optimizer = torch.optim.AdamW(swin_unetr.parameters(), lr=0.001, weight_decay=0.00001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        return swin_unetr, criterion, optimizer, scheduler
