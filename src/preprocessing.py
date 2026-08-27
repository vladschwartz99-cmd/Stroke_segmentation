import nibabel
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    CropForegroundd,  RandCropByPosNegLabeld, SpatialPadd, ConcatItemsd,
    DeleteItemsd, ScaleIntensityRangePercentilesd, RandRotated, RandFlipd,
    RandGaussianNoised, RandAdjustContrastd, CastToTyped, EnsureTyped
)
from scipy.ndimage import label
from src.data_utils import build_dataframe_to_split



def split_dataset(random_state=42, to_ensemble=False):
    """Функция, разбивающая датасет на train, val и test со стратификацией по
                id пациента и группе количества и объема очагов"""

    # Получение датафрейма с признаками, необходимыми для разбиения со стратификацией
    df_to_split = build_dataframe_to_split(to_ensemble=to_ensemble)

    # Поскольку StratifiedGroupKFold плохо работает с малыми классами,
    # отделим их и позже добавим вручную
    no_lesion_group = df_to_split[df_to_split['lesion_label'] == 'no_lesion']
    multiple_only_small_group = df_to_split[df_to_split['lesion_label'] == 'multiple_only_small_lesion']
    large_groups = df_to_split[~df_to_split['lesion_label'].isin(['no_lesion', 'multiple_only_small_lesion'])]

    # Разбиваем данные на train и test_val
    splitter_1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_ids, test_val_ids = next(splitter_1.split(X=large_groups, y=large_groups['lesion_label'], groups=large_groups['patient_id']))

    test_val_df = large_groups.iloc[test_val_ids].reset_index(drop=True)

    # Разбиваем данные на val и test
    splitter_2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=random_state)
    val_ids, test_ids = next(splitter_2.split(X=test_val_df, y=test_val_df['lesion_label'], groups=test_val_df['patient_id']))

    # Формирование датафреймов подвыборок
    train_df = large_groups.iloc[train_ids]
    val_df = test_val_df.iloc[val_ids]
    test_df = test_val_df.iloc[test_ids]

    # Получение id пациентов для малых групп
    no_lesion_group_ids = no_lesion_group['patient_id'].unique()
    multiple_only_small_group_ids = multiple_only_small_group['patient_id'].unique()

    # Добавление пациентов из малых групп в train
    train_df = pd.concat(
        [train_df,
         no_lesion_group[no_lesion_group['patient_id'] == no_lesion_group_ids[0]],
         multiple_only_small_group[multiple_only_small_group['patient_id'] == multiple_only_small_group_ids[0]]
        ],
        ignore_index=True
    )

    # Добавление пациентов из малых групп в val
    val_df = pd.concat(
        [val_df,
         no_lesion_group[no_lesion_group['patient_id'] == no_lesion_group_ids[1]]
        ],
        ignore_index=True
    )

    # Добавление пациентов из малых групп в test
    test_df = pd.concat(
        [test_df,
         no_lesion_group[no_lesion_group['patient_id'] == no_lesion_group_ids[2]],
         multiple_only_small_group[multiple_only_small_group['patient_id'] == multiple_only_small_group_ids[1]]
        ],
        ignore_index=True
    )

    return train_df, val_df, test_df



def get_transforms(protocols_list, patch_size=(96, 96, 96), augmentations=False):
    """Функция, формирующая протокол трансформаций train и val/test для заданного количества изображений"""

    # Добавление маски к списку протоколов
    protocols_mask_list = protocols_list + ['mask']

    if augmentations:

        train_transforms = Compose([

            # Загрузка изображения и маски
            LoadImaged(keys=protocols_mask_list),

            # Приведение тензоров в необходимый формат
            EnsureChannelFirstd(keys=protocols_mask_list),

            # Приведение к единому пространственному положению
            Orientationd(keys=protocols_mask_list, axcodes='RAS', labels=None),

            # Приведение к единому размеру вокселя
            Spacingd(
                keys=protocols_mask_list,
                pixdim=(1.0, 1.0, 1.0),
                mode=['bilinear' if k in protocols_list else 'nearest' for k in protocols_mask_list]
            ),

            # Удаление фона для получения более информативных изображений
            CropForegroundd(
                keys=protocols_mask_list, source_key='dwi',
                select_fn=lambda x: x > 0, margin=5,
                allow_smaller=True
            ),

            # Нормализация, с отсеиванием экстремальных значений для подавления шума, каждого
            # МР-изображения по отдельности
            *[ScaleIntensityRangePercentilesd(
                keys=[protocol], lower=1, upper=99,
                b_min=0.0, b_max=1.0, clip=True
            ) for protocol in protocols_list],

            # Объединение протоколов в одно многоканальное изображение
            ConcatItemsd(keys=protocols_list, name='image', dim=0),

            # Удаляем ненужные ключи
            DeleteItemsd(keys=protocols_list),

            # Патчинг со сдвигом в сторону очагов инсульта для борьбы с дисбалансом классов
            RandCropByPosNegLabeld(
                keys=['image', 'mask'], label_key='mask', spatial_size=patch_size,
                pos=3, neg=1, num_samples=1
            ),

            # Паддинг изображения, в случае если размер изображения не позволил вырезать патч нужного размера
            SpatialPadd(
                keys=['image', 'mask'], spatial_size=patch_size,
                mode=('constant', 'constant'), value=(0, 0)
            ),

            # Аугментации
            # Поворот изображения
            RandRotated(
                keys=['image', 'mask'], range_x=0.26, range_y=0.26,
                range_z=0.26, prob=0.5, mode=('bilinear', 'nearest')
            ),

            # Отражение по оси X
            RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=0),

            # Отражение по оси Y
            RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=1),

            # Отражение по оси Z
            RandFlipd(keys=['image', 'mask'], prob=0.5, spatial_axis=2),

            # Добавление шума
            RandGaussianNoised(keys=['image'], prob=0.3, mean=0.0, std=0.05),

            # Изменение контраста
            RandAdjustContrastd(keys=['image'], prob=0.4, gamma=(0.5, 2.0)),

            # Приведение изображений к нужному формату данных
            CastToTyped(keys=['image'], dtype='float32'),

            # Приведение масок к нужному формату данных
            CastToTyped(keys=['mask'], dtype='uint8')
        ])

        return train_transforms

    else:

        test_transforms = Compose([

            # Загрузка изображения и маски
            LoadImaged(keys=protocols_mask_list),

            # Приведение тензоров в необходимый формат
            EnsureChannelFirstd(keys=protocols_mask_list),

            # Приведение к единому пространственному положению
            Orientationd(keys=protocols_mask_list, axcodes='RAS', labels=None),

            # Приведение к единому размеру вокселя
            Spacingd(
                keys=protocols_mask_list,
                pixdim=(1.0, 1.0, 1.0),
                mode=['bilinear' if k in protocols_list else 'nearest' for k in protocols_mask_list]
            ),

            # Удаление фона для получения более информативных изображений
            CropForegroundd(
                keys=protocols_mask_list, source_key='dwi',
                select_fn=lambda x: x > 0, margin=5,
                allow_smaller=True
            ),

            # Нормализация, с отсеиванием экстремальных значений для подавления шума, каждого
            # МР-изображения по отдельности
            *[ScaleIntensityRangePercentilesd(
                keys=[protocol], lower=1, upper=99,
                b_min=0.0, b_max=1.0, clip=True
            ) for protocol in protocols_list],

            # Объединение протоколов в одно многоканальное изображение
            ConcatItemsd(keys=protocols_list, name='image', dim=0),

            # Удаляем ненужные ключи
            DeleteItemsd(keys=protocols_list),

            # Приведение изображений и масок к нужному формату данных
            EnsureTyped(keys=['image', 'mask'])
        ])

        return test_transforms



class MRIDataset(Dataset):
    """Класс для формирования пар предобработанных изображения и маски"""

    def __init__(self, subset_df, protocols_list, mask_label='mask', patch_size=(96, 96, 96), augmentations=False, save_transforms_meta=False):
        self.patient_ids = subset_df['patient_id'].unique()
        self.flair_df = subset_df[subset_df['label'] == 'flair']
        self.adc_df = subset_df[subset_df['label'] == 'adc']
        self.dwi_df = subset_df[subset_df['label'] == 'dwi']
        self.masks_df = subset_df[subset_df['label'] == mask_label]
        self.augmentations = augmentations
        self.mask_label = mask_label
        self.protocols_list = protocols_list
        self.transformer = get_transforms(protocols_list=self.protocols_list, patch_size=patch_size, augmentations=self.augmentations)
        self.save_transforms_meta = save_transforms_meta


    def __getitem__(self, idx):

        # Получение id пациента
        patient_id = self.patient_ids[idx]

        # Создание словаря для добавления путей к изображениям
        img_mask_dict = {}

        # Добавление пути к изображению определенного протокола, если он указан в списке протоколов
        for protocol, protocol_df in zip(['flair', 'dwi', 'adc'], [self.flair_df, self.dwi_df, self.adc_df]):
            if protocol in self.protocols_list:
                protocol_row = protocol_df[protocol_df['patient_id'] == patient_id].iloc[0]
                img_mask_dict[protocol] = protocol_row['file_path']

        # Добавление пути к маске
        mask_row = self.masks_df[self.masks_df['patient_id'] == patient_id].iloc[0]
        img_mask_dict['mask'] = mask_row['file_path']

        # Трансформация изображений и маски
        transformed = self.transformer(img_mask_dict)

        # Извлечение словаря, в случае если он упаковался в список
        if isinstance(transformed, list):
            transformed = transformed[0]

        if self.save_transforms_meta:
            return transformed

        return transformed['image'], transformed['mask']


    def __len__(self):
        return len(self.patient_ids)



def build_loader(
        subset_df, protocols_list, mask_label='mask', patch_size=(96, 96, 96),
        augmentations=False, batch_size=8, shuffle=False,
        save_transforms_meta=False
):
    """Функция, формирующая loader для подвыборки"""

    dataset = MRIDataset(
        subset_df, protocols_list=protocols_list, mask_label=mask_label, patch_size=patch_size,
        augmentations=augmentations, save_transforms_meta=save_transforms_meta
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader



def split_mask(mask_path, patient_id):
    """Функция, разбивающая маску исходной сегментации на маски крупных и мелких очагов"""

    # Загрузка и получение содержимого маски
    mask_img = nibabel.load(mask_path)
    full_mask = mask_img.get_fdata()

    # Получение объема вокселя маски и матрицы афинных преобразований
    metadata = mask_img.header
    voxel_volume = math.prod(metadata['pixdim'][1:4])
    affine = mask_img.affine

    # Выделение отдельных очагов
    labels, n = label(full_mask)

    # Создание пустых масок размера исходной
    large_mask = np.zeros_like(full_mask)
    small_mask = np.zeros_like(full_mask)

    # Вычисление объема каждого очага и добавление в маску
    for lesion_id in range(1, n + 1):

        # Получение маски очага и вычисление его объема
        lesion = labels == lesion_id
        volume = lesion.sum() * voxel_volume

        # Добавление очага в соответствующую маску
        if volume > 44.11:
            large_mask[lesion] = 1
        else:
            small_mask[lesion] = 1

    # Преобразуем новые маски в NifTi
    large_mask = nibabel.Nifti1Image(large_mask, affine)
    small_mask = nibabel.Nifti1Image(small_mask, affine)

    # Формируем пути сохранения новых масок
    output_dir = mask_path.parent
    large_output_path = output_dir / f'{patient_id}_large_mask.nii'
    small_output_path = output_dir / f'{patient_id}_small_mask.nii'

    # Сохраняем новые маски, рядом с исходной
    nibabel.save(large_mask, large_output_path)
    nibabel.save(small_mask, small_output_path)
