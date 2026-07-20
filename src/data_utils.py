import kagglehub
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import nibabel
import math
import matplotlib.pyplot as plt
from nilearn import plotting
from nibabel.processing import resample_from_to
from scipy.ndimage import label



def download_data():
    """Функция, скачивающая данные и возвращающая путь к ним"""

    # Скачивание данных
    dataset_path = kagglehub.dataset_download('orvile/isles-2022-brain-stoke-dataset')

    # Добавление подпапок в общий путь
    dataset_path = Path(dataset_path) / 'ISLES-2022' / 'ISLES-2022'

    return dataset_path



def count_file_type(dataset_path):
    """Функция, подсчитывающая количество файлов по типам расширения"""

    # Словарь для подсчета количества фалов
    extension_counts = defaultdict(int)

    # Обход всех вложенных папок
    for path in dataset_path.rglob('*'):
        if path.is_file():
            # Получение расширения файла
            ext = path.suffix.lower()

            # Учет файлов без расширения
            if not ext:
                ext = 'without_extension'

            # Увеличение количества в словаре
            extension_counts[ext] += 1

    return extension_counts



def build_paths_df(dataset_path):
    """Функция, формирующая датафрейм с полными путями к МР-изображениям и маскам,
                    ID пациентов и протоколами МРТ/метками маски"""

    # Список для сбора данных о файлах
    data_list = []

    # Сбор полных путей всех файлов .nii
    nii_files = [nii for nii in dataset_path.rglob('*.nii*') if nii.is_file()]

    # Получение дополнительных данных для каждого снимка
    for file in nii_files:

        # Условие для выявления масок
        if file.parents[2].name == 'derivatives':
            patient_id = int(file.parents[1].name[-4:])
            label = 'mask'

        # Условие для выявления FLAIR изображений
        elif file.parent.name == 'anat':
            patient_id = int(file.parents[2].name[-4:])
            label = 'flair'

        # Остальные случаи, так как структура пути к DWI и ADC разнится для некоторых файлов
        else:
            patient_id = int(file.parents[3].name[-4:])

            # Обновление id пациента, если снимок не вложен в отдельную папку
            if patient_id == 2022:
                patient_id = int(file.parents[2].name[-4:])

            # К счастью протокол (в отличие от id пациента) указан в названии каждого снимка
            label = 'adc' if 'adc' in file.name else 'dwi'

        # Добавление в общий список
        data_list.append({
            'file_path': file,
            'patient_id': patient_id,
            'label': label,
        })

    # Создаем датафрейм
    df = pd.DataFrame(data_list, columns=['file_path', 'patient_id', 'label'])

    return df



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



def visualize_all_patient_images(patient_df):
    """Функция визуализации МР-изображений и маски сегментации"""

    fig, ax = plt.subplots(2, 2, figsize=(18, 10))

    # Вывод всех снимков и маски в трех проекциях
    for axes, label in zip(ax.flatten(), patient_df['label'].unique()):
        plotting.plot_anat(patient_df[patient_df['label'] == label]['file_path'].item(),
                           display_mode="ortho", title=label, axes=axes
        )

    fig.suptitle('Пример маски и всех МР-изображений для одного пациента')
    plt.show()



def visualize_mri_with_mask(mri_df, mask_df):
    """Функция визуализации МР-изображений с наложением маски сегментации"""

    fig, ax = plt.subplots(2, 2, figsize=(18, 10))

    # Вывод всех снимков с наложенной маской в трех проекциях
    for axes, label in zip(ax.flatten(), mri_df['label'].unique()):
        plotting.plot_roi(
            mask_df['file_path'].item(),
            mri_df[mri_df['label'] == label]['file_path'].item(),
            display_mode="ortho", title=label, axes=axes, alpha=0.2
        )

    fig.suptitle('Пример всех МР-изображений с наложенной маской для одного пациента')
    plt.show()



def visualize_one_slice(file_path, coordinate, slice_idx):
    """Функция визуализирующая заданный срез по заданной оси для одного изображения"""

    fig = plt.figure(figsize=(5, 5))

    plotting.plot_anat(
        file_path, display_mode=coordinate,
        cut_coords=[slice_idx], figure=fig
    )

    plt.show()



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

        # Вычисление объема маски в см3
        total_volume_cm3 = voxel_count * row.voxel_volume_mm3 / 1000

        # Подсчет количества очагов
        labels, n_lesions = label(mask_data)

        # Отсеивание очагов объемом менее 50 мм3
        real_lesions = 0

        for i in range(1, n_lesions + 1):
            size = np.sum(labels == i)
            volume = size * row.voxel_volume_mm3

            if volume >= 50:
                real_lesions += 1

        # Добавление в общий список
        volumes_list.append({
            'file_path': row.file_path,
            'total_volume_cm3': total_volume_cm3,
            'n_lesions': real_lesions if real_lesions > 0 else n_lesions
        })

    # Формирование датафрейма новых признаков
    new_mask_df = pd.DataFrame(volumes_list, columns=['file_path', 'total_volume_cm3', 'n_lesions'])

    # Объединение датафреймов
    full_mask_df = pd.merge(masks_df, new_mask_df, on='file_path')

    return full_mask_df



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



def add_label(row):
    """Функция для присвоения метки класса на основе количества очагов и их объема"""

    # Условие для пустой маски сегментации
    if row['total_volume_cm3'] == 0:
        return 'no_lesion'

    # Условие для одного малого очага (с сохранением пациентов у которых есть очаг меньше 50 мм3)
    elif row['n_lesions'] <= 1 and row['n_large_lesions'] == 0 and row['total_volume_cm3'] != 0:
        return 'one_small_lesion'

    # Условие для одного крупного очага
    elif row['n_lesions'] == 1 and row['n_large_lesions'] == 1:
        return 'one_large_lesion'

    # Условие для множественных только малых очагов
    elif row['n_lesions'] > 1 and row['n_large_lesions'] == 0:
        return 'multiple_only_small_lesion'

    # Условие для множественных только крупных очагов
    elif row['n_lesions'] > 1 and row['n_small_lesions'] == 0:
        return 'multiple_only_large_lesion'

    # Остальные случаи это множественные очаги разного объема
    else:
        return 'multiple_mixed_lesion'



def build_dataframe_to_split():
    """Функция, формирующая датафрейм из путей к изображениям и признаков,
                необходимых для разбиения на подвыборки"""

    # Получение пути к датафрейму (и загрузка при его отсутствии)
    dataset_path = download_data()

    # Формирование датафрейма из пути к файлу, id пациента и маркера протокола/маски
    paths_df = build_paths_df(dataset_path)

    # Получение датафрейма масок сегментации
    masks_df = paths_df[paths_df['label'] == 'mask']

    # Получения количества и объема очагов каждого пациента
    lesions_list = []

    for row in masks_df.itertuples():

        # Загрузка и получение содержимого масок и метаданных
        mask_img = nibabel.load(row.file_path)
        mask_data = mask_img.get_fdata()
        metadata = mask_img.header

        # Вычисление объема вокселя
        voxel_volume_mm3 = math.prod(metadata['pixdim'][1:4])

        # Подсчет количества вокселей
        voxel_count = np.sum(mask_data > 0)

        # Вычисление объема маски в см^3
        total_volume_cm3 = voxel_count * voxel_volume_mm3 / 1000

        # Подсчет количества очагов
        labels, n_lesions = label(mask_data)

        # Отсеиваем очаги объемом менее 50 мм3 и подсчитываем
        # количество крупных и мелких очагов по порогу в 30 см3
        real_lesions = 0
        small_lesions = 0
        large_lesions = 0

        for i in range(1, n_lesions + 1):
            size = np.sum(labels == i)
            volume = size * voxel_volume_mm3

            if volume >= 50:
                real_lesions += 1

                if volume < 30000:
                    small_lesions += 1
                else:
                    large_lesions += 1

        # Добавление в общий список
        lesions_list.append({
            'patient_id': row.patient_id,
            'total_volume_cm3': total_volume_cm3,
            'n_lesions': real_lesions,
            'n_small_lesions': small_lesions,
            'n_large_lesions': large_lesions
        })

    # Формирование датафрейма новых признаков
    lesions_df = pd.DataFrame(
        lesions_list,
        columns=[
            'patient_id', 'total_volume_cm3',
            'n_lesions', 'n_small_lesions',
            'n_large_lesions'
        ]
    )

    # Преобразование количества и объема очагов в метку группы
    lesions_df['lesion_label'] = lesions_df.apply(add_label, axis=1)

    # Присоединение метки к основному датафрейму
    df_to_split = paths_df.merge(lesions_df, on='patient_id', how='left')

    return df_to_split
