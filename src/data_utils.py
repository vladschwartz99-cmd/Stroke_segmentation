import kagglehub
import math
import nibabel
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.ndimage import label


def download_data():
    """Функция, скачивающая данные и возвращающая путь к ним"""

    # Скачивание данных
    dataset_path = kagglehub.dataset_download(
        'orvile/isles-2022-brain-stoke-dataset',
        output_dir=Path(r'C:\project_data')
    )

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



def build_paths_df_to_split(dataset_path):
    """Функция, формирующая датафрейм с полными путями к МР-изображениям после регистрации
                    и маскам, ID пациентов и протоколами МРТ/метками маски"""

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

        # Условие для выявления регистрированных FLAIR изображений
        elif file.parent.name == 'anat' and 'register' in file.name:
            patient_id = int(file.parents[2].name[-4:])
            label = 'flair'

        # Пропуск исходных FLAIR
        elif file.parent.name == 'anat' and 'register' not in file.name:
            continue

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



def add_label(row):
    """Функция для присвоения метки класса на основе количества очагов и их объема"""

    # Условие для пустой маски сегментации
    if row['n_lesions'] == 0:
        return 'no_lesion'

    # Условие для одного малого очага
    elif row['n_lesions'] == 1 and row['n_small_lesions'] == 1:
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
    paths_df = build_paths_df_to_split(dataset_path)

    # Получение датафрейма масок сегментации
    masks_df = paths_df[paths_df['label'] == 'mask']

    # Получения количества и объема очагов каждого пациента
    lesions_list = []

    for row in masks_df.itertuples():

        # Загрузка и получение содержимого масок
        mask_img = nibabel.load(row.file_path)
        mask_data = mask_img.get_fdata()
        metadata = mask_img.header

        # Подсчет количества очагов
        labels, n_lesions = label(mask_data)

        # Подсчитываем количество крупных и мелких очагов по порогу равному медиане объема
        all_lesions = 0
        small_lesions = 0
        large_lesions = 0

        for i in range(1, n_lesions + 1):

            # Подсчитываем объем очага
            size = np.sum(labels == i)
            volume = size * math.prod(metadata['pixdim'][1:4])

            # Обновляем счетчики
            all_lesions += 1

            if volume < 44.11:
                small_lesions += 1
            else:
                large_lesions += 1

        # Добавление в общий список
        lesions_list.append({
            'patient_id': row.patient_id,
            'n_lesions': all_lesions,
            'n_small_lesions': small_lesions,
            'n_large_lesions': large_lesions
        })

    # Формирование датафрейма новых признаков
    lesions_df = pd.DataFrame(
        lesions_list,
        columns=[
            'patient_id', 'n_lesions',
            'n_small_lesions', 'n_large_lesions'
        ]
    )

    # Преобразование количества и объема очагов в метку группы
    lesions_df['lesion_label'] = lesions_df.apply(add_label, axis=1)

    # Присоединение метки к основному датафрейму
    df_to_split = paths_df.merge(lesions_df, on='patient_id', how='left')

    return df_to_split
