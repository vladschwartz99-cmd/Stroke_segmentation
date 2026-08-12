import nibabel
import math
import pandas as pd
import numpy as np
from nibabel.processing import resample_from_to



def enrich_dataframe(df):
    """Функция, добавляющая в датафрейм дополнительные признаки из метаданных .nii файлов"""

    # Список для сохранения новых признаков
    new_features = []

    # Получение признаков для каждого изображения
    for row in df.itertuples():

        # Путь к изображению
        file_path = row.file_path

        # Получение метаданных
        img = nibabel.load(file_path)
        metadata = img.header

        # Получение новых признаков и добавление в список
        new_features.append({
            'file_path': file_path,
            'image_dimension': metadata['dim'][0],
            'image_X_size': metadata['dim'][1],
            'image_Y_size': metadata['dim'][2],
            'image_Z_size': metadata['dim'][3],
            'voxel_volume_mm3': math.prod(metadata['pixdim'][1:4]),
            'affine_matrix': tuple(map(tuple, img.affine)),
            'data_type': metadata.get_data_dtype().name
        })

    # Формирование датафрейма новых признаков
    new_features_df = pd.DataFrame(
        new_features,
        columns=[
            'file_path', 'image_dimension', 'image_X_size', 'image_Y_size',
            'image_Z_size', 'voxel_volume_mm3', 'affine_matrix', 'data_type'
        ]
    )

    # Объединение датафреймов
    full_df = pd.merge(df, new_features_df, on='file_path')

    return full_df



def intensity_statistics(df):
    """Функция, подсчитывающая характеристики распределения интенсивности
    изображения внутри границ маски (при ее наложении) и за ее пределами"""

    # Формируем датафреймы из МР-изображений и масок
    mri_df = df[df['label'] != 'mask']
    mask_df = df[df['label'] == 'mask']

    # Список для сохранения результата
    intensity_list = []

    for mri in mri_df.itertuples():

        # Получение пути к маске
        mask_path = mask_df[mask_df['patient_id'] == mri.patient_id]

        # Загрузка МР-изображения
        img = nibabel.load(mri.file_path)
        image = img.get_fdata()

        # Загрузка маски
        mask = nibabel.load(mask_path['file_path'].item())
        lesion_mask = mask.get_fdata().astype(bool)

        # Трансформация в случае несоответствия размеров изображения и маски
        if lesion_mask.shape != image.shape:

            # Приведение маски к размеру изображения
            resample_mask = resample_from_to(mask, img, order=0)

            # Пересохранение маски
            lesion_mask = resample_mask.get_fdata().astype(bool)

        # Получение маски для всего мозга
        brain_mask = image > 0

        # Получение инвертированной маски мозга без очагов
        outside_mask = brain_mask & (~lesion_mask)

        # Извлечение интенсивностей
        intensities_inside = image[lesion_mask]
        intensities_outside = image[outside_mask]
        intensities_brain = image[brain_mask]

        # словарь из только интенсивности всего мозга для пациентов с пустыми масками
        if len(intensities_inside) == 0:
            zones = {
                'full_brain': intensities_brain
            }

        # Формирование словаря масок разных зон
        else:
            zones = {
                'inside_mask': intensities_inside,
                'outside_mask': intensities_outside,
                'full_brain': intensities_brain
            }

        # Добавление в список характеристик распределения интенсивности
        for name, data in zones.items():
            intensity_list.append({
                'label': mri.label,
                'zone': name,
                'volume_cm3': (len(data) * mri.voxel_volume_mm3) / 1000,
                'min': np.min(data),
                'max': np.max(data),
                'mean': np.mean(data),
                'median': np.median(data),
                'std': np.std(data),
                'percentile_25': np.percentile(data, 25),
                'percentile_75': np.percentile(data, 75),
            })

    # Преобразование в датафрейм
    intensity_df = pd.DataFrame(
        intensity_list,
        columns=[
            'label', 'zone', 'volume_cm3', 'min', 'max', 'mean',
            'median', 'std', 'percentile_25', 'percentile_75'
        ]
    )

    return intensity_df
