import nibabel
import numpy as np
import pandas as pd
from scipy.ndimage import label



def lesion_volume_quartiles(full_df):
    """Функция, вычисляющая квартили распределения очагов по объему"""

    # Формирование датасета из масок
    masks_df = full_df[full_df['label'] == 'mask']

    # Список для сбора объемов
    lesion_volumes = []

    # Для каждой маски
    for row in masks_df.itertuples():

        # Загрузка и получение содержимого масок
        mask_img = nibabel.load(row.file_path)
        mask_data = mask_img.get_fdata()

        # Выделение отдельных очагов
        labels, n = label(mask_data)

        # Вычисление объема каждого очага и добавление в список
        for lesion_id in range(1, n + 1):

            lesion = labels == lesion_id

            volume = lesion.sum() * row.voxel_volume_mm3

            lesion_volumes.append(volume)

    # Вычисление медианы
    median = np.median(lesion_volumes)

    return lesion_volumes, median



def mask_features(full_df):
    """Функция по извлечению характеристик ишемического поражения из масок сегментации"""

    # Формирование датасета из масок
    masks_df = full_df[full_df['label'] == 'mask']

    # Список для сохранения результатов
    volumes_list = []

    # Получения объема и количества очагов каждой маски
    for row in masks_df.itertuples():

        # Загрузка и получение содержимого масок
        mask_img = nibabel.load(row.file_path)
        mask_data = mask_img.get_fdata()

        # Подсчет количества вокселей
        voxel_count = np.sum(mask_data > 0)

        # Вычисление общего объема маски в см3
        total_volume_cm3 = voxel_count * row.voxel_volume_mm3 / 1000

        # Подсчет количества очагов
        labels, n_lesions = label(mask_data)

        # Добавление в общий список
        volumes_list.append({
            'file_path': row.file_path,
            'total_volume_cm3': total_volume_cm3,
            'n_lesions': n_lesions
        })

    # Формирование датафрейма новых признаков
    new_mask_df = pd.DataFrame(volumes_list, columns=['file_path', 'total_volume_cm3', 'n_lesions'])

    # Объединение датафреймов
    full_mask_df = pd.merge(masks_df, new_mask_df, on='file_path')

    return full_mask_df
