import torch
import numpy as np
import pandas as pd
from monai.inferers import sliding_window_inference
from scipy.ndimage import label, binary_opening
from scipy.stats import spearmanr
from medpy.metric.binary import hd95
from src.preprocessing import build_loader


# Устройство для вычислений
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_prediction(image, model):
    """Функция формирующая маску сегментации на основе предсказания модели"""

    # Перенос изображения на устройство для вычислений
    image = image.to(device)

    model.eval()
    with torch.no_grad():

        # Патчинг изображения и получение ответа модели
        logits = sliding_window_inference(
            inputs=image, roi_size=(96, 96, 96),
            sw_batch_size=1, predictor=model,
            overlap=0.5, mode='gaussian'
        )

    # Преобразование в маску вероятностей
    pred_mask = torch.sigmoid(logits)

    return pred_mask



def metrics_for_lesion(pred_lesion_mask, true_lesion_mask):
    """Функция, рассчитывающая Dice, Iou, HD для предсказанной и истинной маски"""

    # Приведение масок к булевому типу
    pred_mask = pred_lesion_mask.astype(bool)
    true_mask = true_lesion_mask.astype(bool)

    # Снижение размерности
    pred_mask = np.squeeze(pred_mask)
    true_mask = np.squeeze(true_mask)

    # Подсчет количества ненулевых пикселей
    pred_vol, true_vol = np.sum(pred_mask), np.sum(true_mask)

    # В случае отсутствия поражения и ложной сегментации
    if pred_vol == 0 and true_vol == 0:

        dice, iou, hd = 1.0, 1.0, 0.0

        return dice, iou, hd

    # В случае ложноположительной или ложноотрицательной сегментации моделью
    if pred_vol == 0 or true_vol == 0:

        dice, iou, hd = 0.0, 0.0, np.nan

        return dice, iou, hd

    # Расчет пересечения
    intersection = np.sum(pred_mask & true_mask)

    # Общая площадь
    union = np.sum(pred_mask | true_mask)

    dice = (2.0 * intersection) / (pred_vol + true_vol) if pred_vol + true_vol > 0 else 0.0
    iou = intersection / union if union > 0 else 0.0

    # Вычисление расстояния Хаусдорфа
    hd = hd95(pred_mask, true_mask)

    return dice, iou, hd



def get_overlaps(pred_mask, true_mask):
    """Функция расчета доли пересечения истинной и предсказанной масок"""

    # Приведение масок к булевому типу
    pred_mask = np.asarray(pred_mask).astype(bool)
    true_mask = np.asarray(true_mask).astype(bool)

    # Снижение размерности
    pred_mask = np.squeeze(pred_mask)
    true_mask = np.squeeze(true_mask)

    # Получение объема пересечения
    intersection = np.sum(true_mask & pred_mask)

    # Расчет доли пересечения предсказанной маски с истинной
    pred_volume = np.sum(pred_mask)
    pred_overlap = intersection / pred_volume if pred_volume > 0 else 0

    # Расчет доли пересечения истинной маски с предсказанной
    true_volume = np.sum(true_mask)
    true_overlap = intersection / true_volume if true_volume > 0 else 0

    return pred_overlap, true_overlap




def global_metrics(pred_true_masks, threshold=0.5):
    """Функция, рассчитывающая метрики на уровне целых снимков"""

    # Метрики для подсчета
    tp, fp, fn = 0, 0, 0
    dice_list, iou_list, hd_list = [], [], []

    # Списки для сохранения объемов исходной и предсказанной сегментации
    true_volumes = []
    pred_volumes = []

    for pred_mask_raw , true_mask_raw in pred_true_masks:

        # Преобразование предсказанной маски в бинарную на основе порога уверенности
        pred_mask_raw = pred_mask_raw > threshold

        # Морфологическое открытие масок и его отмена
        # при удалении сегментации
        true_mask = binary_opening(true_mask_raw)
        if np.sum(true_mask) == 0:
            true_mask = true_mask_raw

        pred_mask = binary_opening(pred_mask_raw)
        if np.sum(pred_mask) == 0:
            pred_mask = pred_mask_raw

        # Добавление объемов поражения исходной и предсказанной маски в списки
        true_volumes.append(np.sum(true_mask))
        pred_volumes.append(np.sum(pred_mask))

        # Получение метрик соответствия очагов
        dice, iou, hd = metrics_for_lesion(pred_mask, true_mask)

        # Добавление метрик в списки
        dice_list.append(dice)
        iou_list.append(iou)
        hd_list.append(hd)

        # Расчет доли пересечения истинной и предсказанной масок
        pred_overlap, true_overlap = get_overlaps(pred_mask, true_mask)

        # Если маски пересекаются (с небольшим порогом из-за большого количества мелких очагов)
        if true_overlap >= 0.15 and pred_overlap >= 0.15:
            tp +=1

        # Если маски не пересекаются / слабо пересекаются
        else:

            # Если истинная маска не пуста
            if np.any(true_mask):
                fn +=1

            # Если предсказанная маска не пуста
            if np.any(pred_mask):
                fp +=1

    # Расчет recall, precision и f1
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    # Удаление Nan из списка HD
    valid_hd = [x for x in hd_list if not np.isnan(x)]

    # Расчет корреляции Спирмена
    spearman_coef, spearman_p_value = spearmanr(true_volumes, pred_volumes)

    # Формирование словаря метрик
    metrics_dict = {
        'threshold': threshold,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'dice': np.mean(dice_list) if dice_list else 0.0,
        'iou': np.mean(iou_list) if iou_list else 0.0,
        'hd': np.mean(valid_hd) if valid_hd else 0.0,
        'spearman_coef': spearman_coef,
        'spearman_p_value': spearman_p_value
    }

    return metrics_dict



def find_optim_threshold(model, test_loader):
    """Функция, подсчитывающая глобальные метрики для разных порогов уверенности модели"""

    thresholds_metrics_list = []

    pred_true_masks = []

    # Получение предсказаний модели
    for image, true_mask in test_loader:

        pred_mask = get_prediction(image, model)

        # Добавление в список пар предсказанных и истинных масок
        pred_true_masks.append(
            (pred_mask.cpu().detach().numpy(),
            true_mask.cpu().detach().numpy())
        )


    for threshold in np.arange(0.1, 1, 0.05):

        # Получение метрик для порога уверенности
        metrics_dict = global_metrics(pred_true_masks, threshold=threshold)

        # Добавление метрик в общий список
        thresholds_metrics_list.append(metrics_dict)

    # Преобразование в датафрейм
    thresholds_metrics = pd.DataFrame(
        thresholds_metrics_list,
        columns=[
            'threshold', 'recall', 'precision', 'f1', 'dice',
            'iou', 'hd', 'spearman_coef', 'spearman_p_value'
        ]
    )

    # Находим строку максимального значения Dice
    thresholds_metrics = thresholds_metrics.loc[thresholds_metrics['dice'].idxmax()]

    # Получаем значение лучшего порога уверенности
    best_threshold = thresholds_metrics['threshold'].item()

    return thresholds_metrics.to_frame().T, best_threshold




def groups_metrics_reports(model, patients_df, protocols_list, threshold=0.5):
    """Функция, рассчитывающая метрики сегментации по группам пациентов и по объему очагов"""

    # Список для сохранения итоговых метрик по группам
    groups_metrics = []

    # Получение названий групп пациентов
    patients_groups = patients_df['lesion_label'].unique()

    patients_dice = {}

    # Для каждой группы
    for group in patients_groups:

        # Формирование подвыборки из пациентов определенной группы
        group_df = patients_df[patients_df['lesion_label'] == group]

        # Получение количества пациентов в группе
        n_patients = len(group_df['patient_id'].unique())

        # Сохранение id пациентов
        patients_ids = group_df['patient_id'].drop_duplicates().tolist()

        # Преобразование в лоадер
        group_loader = build_loader(
            group_df, protocols_list, augmentations=False,
            batch_size=1, shuffle=False
        )

        # Переменные для подсчета метрик по изображению
        tp, fp, fn = 0, 0, 0

        # Списки для сохранения объемов исходной и предсказанной сегментации
        true_volumes = []
        pred_volumes = []

        # Список для сохранения метрик по пациентам
        metrics_for_patient = []

        # Для каждого изображения
        for patient_id, (image, true_mask_raw) in zip(patients_ids, group_loader):

            # Получение предсказанной маски
            pred_mask_raw = get_prediction(image, model)

            # Преобразование в бинарную маску по порогу
            pred_mask_raw = pred_mask_raw > threshold

            # Морфологическое открытие масок и его отмена
            # при удалении сегментации
            true_mask = binary_opening(true_mask_raw)
            if np.sum(true_mask) == 0:
                true_mask = true_mask_raw

            pred_mask = binary_opening(pred_mask_raw.cpu())
            if np.sum(pred_mask) == 0:
                pred_mask = pred_mask_raw

            # Перевод масок на cpu
            pred_mask, true_mask = pred_mask.to('cpu'), true_mask.to('cpu')

            # Подсчет количества очагов
            pred_labels, pred_n_lesions = label(pred_mask)
            true_labels, true_n_lesions = label(true_mask)

            # Расчет доли пересечения истинной и предсказанной масок
            pred_overlap, true_overlap = get_overlaps(pred_mask, true_mask)

            # Если маски пересекаются (с небольшим порогом из-за большого количества мелких очагов)
            if true_overlap >= 0.15 and pred_overlap >= 0.15:
                tp += 1

            # Если пересечение меньше
            else:

                # Если исходная маска не пустая
                if np.any(true_mask):
                    fn += 1

                # Если предсказанная маска не пустая
                if np.any(pred_mask):
                    fp += 1

            # Добавление объемов поражения исходной и предсказанной маски в списки
            true_volumes.append(np.sum(true_mask))
            pred_volumes.append(np.sum(pred_mask))

            # Метрики по очагам для подсчета
            all_lesion_count = 0
            big_lesion_count = 0
            small_lesion_count = 0

            all_dice, big_dice, small_dice = [], [], []
            all_iou, big_iou, small_iou = [], [], []
            all_hd, big_hd, small_hd = [], [], []

            # Расчет метрик для каждого очага исходной разметки
            for true_id in range(1, true_n_lesions + 1):
                true_lesion_mask = (true_labels == true_id)

                # Проверяем, какие ID предсказаний пересекаются с текущим реальным очагом
                overlapping_pred_ids = np.unique(pred_labels[true_lesion_mask])
                overlapping_pred_ids = overlapping_pred_ids[overlapping_pred_ids > 0]

                # Если модель нашла истинный очаг
                if len(overlapping_pred_ids) > 0:

                    # Объединяем все пересекающиеся предсказания,
                    # если модель разбила один реальный очаг на несколько
                    pred_lesion_mask = np.isin(pred_labels, overlapping_pred_ids)

                    # Считаем Dice, IoU, HD для этого очага
                    dice, iou, hd = metrics_for_lesion(pred_lesion_mask, true_lesion_mask)

                    hd_list = []

                    # Расчет расстояния Хаусдорфа для каждого предсказанного очага
                    # если модель для одного истинного очага предсказала несколько
                    for pred_id in overlapping_pred_ids:

                        # Выделение одной компоненты
                        pred_component = pred_labels == pred_id

                        # Расчет расстояния Хаусдорфа
                        hd = hd95(
                            pred_component,
                            true_lesion_mask
                        )

                        # Добавление в общий список
                        hd_list.append(hd)

                    # Обновление счетчиков и списков
                    all_lesion_count += 1
                    all_dice.append(dice)
                    all_iou.append(iou)
                    all_hd.extend(hd_list)

                    # Обновление счетчиков и списков для крупных очагов
                    if np.sum(true_lesion_mask) > 44.11:

                        big_lesion_count += 1
                        big_dice.append(dice)
                        big_iou.append(iou)
                        big_hd.extend(hd_list)

                    # Обновление счетчиков и списков для мелких очагов
                    else:

                        small_lesion_count += 1
                        small_dice.append(dice)
                        small_iou.append(iou)
                        small_hd.extend(hd_list)

            # Удаление Nan из списка HD
            valid_all_hd = [x for x in all_hd if not np.isnan(x)]
            valid_big_hd = [x for x in big_hd if not np.isnan(x)]
            valid_small_hd = [x for x in small_hd if not np.isnan(x)]

            # Сохранение Dice для одного пациента
            patients_dice[patient_id] = np.mean(all_dice) if all_dice else 0.0

            # Добавление метрик в список по пациентам
            metrics_for_patient.append({

                'patient_group': group,

                'all_dice': np.mean(all_dice) if all_dice else 0.0,
                'big_dice': np.mean(big_dice) if big_dice else 0.0,
                'small_dice': np.mean(small_dice) if small_dice else 0.0,

                'all_iou': np.mean(all_iou) if all_iou else 0.0,
                'big_iou': np.mean(big_iou) if big_iou else 0.0,
                'small_iou': np.mean(small_iou) if small_iou else 0.0,

                'all_hd': np.mean(valid_all_hd) if valid_all_hd else 0.0,
                'big_hd': np.mean(valid_big_hd) if valid_big_hd else 0.0,
                'small_hd': np.mean(valid_small_hd) if valid_small_hd else 0.0,
            })

        # Преобразование метрик пациентов группы в датафрейм
        group_metrics_df = pd.DataFrame(
            metrics_for_patient,
            columns=[
                'patient_group',
                'all_dice', 'big_dice', 'small_dice',
                'all_iou', 'big_iou', 'small_iou',
                'all_hd', 'big_hd', 'small_hd'
            ]
        )

        # Вычисление recall
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0

        # Вычисление precision
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0

        # Вычисление f1
        f1 = (
            2 * (precision * recall) /
            (precision + recall)
            if precision + recall > 0 else 0.0
        )

        # Расчет корреляции Спирмена
        if len(true_volumes) > 1:
            spearman_coef, spearman_p_value = spearmanr(true_volumes, pred_volumes)
        else:
            spearman_coef = np.nan
            spearman_p_value = np.nan

        # Добавление средних метрик в общий список
        groups_metrics.append({

            'patient_group': group,
            'patient_count': n_patients,

            'recall': recall,
            'precision': precision,
            'f1': f1,
            'spearman_coef': spearman_coef,
            'spearman_p_value': spearman_p_value,

            'all_dice': group_metrics_df['all_dice'].mean(),
            'big_dice': group_metrics_df['big_dice'].mean(),
            'small_dice': group_metrics_df['small_dice'].mean(),

            'all_dice_std': group_metrics_df['all_dice'].std(),
            'big_dice_std': group_metrics_df['big_dice'].std(),
            'small_dice_std': group_metrics_df['small_dice'].std(),

            'all_iou': group_metrics_df['all_iou'].mean(),
            'big_iou': group_metrics_df['big_iou'].mean(),
            'small_iou': group_metrics_df['small_iou'].mean(),

            'all_iou_std': group_metrics_df['all_iou'].std(),
            'big_iou_std': group_metrics_df['big_iou'].std(),
            'small_iou_std': group_metrics_df['small_iou'].std(),

            'all_hd': group_metrics_df['all_hd'].mean(),
            'big_hd': group_metrics_df['big_hd'].mean(),
            'small_hd': group_metrics_df['small_hd'].mean(),

            'all_hd_std': group_metrics_df['all_hd'].std(),
            'big_hd_std': group_metrics_df['big_hd'].std(),
            'small_hd_std': group_metrics_df['small_hd'].std()
        })

    # Преобразование списка средних метрик в датафрейм
    full_metrics_df = pd.DataFrame(
        groups_metrics,
        columns=[
            'patient_group', 'patient_count',
            'recall', 'precision',  'f1', 'spearman_coef', 'spearman_p_value',
            'all_dice', 'big_dice', 'small_dice',
            'all_dice_std', 'big_dice_std', 'small_dice_std',
            'all_iou', 'big_iou', 'small_iou',
            'all_iou_std', 'big_iou_std', 'small_iou_std',
            'all_hd', 'big_hd', 'small_hd',
            'all_hd_std', 'big_hd_std', 'small_hd_std'
        ]
    )

    # Выделение метрик для очагов всех объемов
    all_lesion_metrics = full_metrics_df[[
        'patient_group', 'patient_count', 'recall',
        'precision', 'f1', 'spearman_coef', 'spearman_p_value',
        'all_dice', 'all_dice_std', 'all_iou', 'all_iou_std',
        'all_hd', 'all_hd_std'
    ]]

    # Выделение метрик для очагов больших объемов
    big_lesion_metrics = full_metrics_df[[
        'patient_group', 'patient_count',
        'big_dice', 'big_dice_std', 'big_iou', 'big_iou_std',
        'big_hd', 'big_hd_std'
    ]]

    # Удаление групп без крупных очагов
    big_lesion_metrics = (
        big_lesion_metrics
        [big_lesion_metrics['patient_group'].isin(
            ['one_large_lesion', 'multiple_mixed_lesion', 'multiple_only_large_lesion'])]
    )

    # Выделение метрик для очагов мелких объемов
    small_lesion_metrics = full_metrics_df[[
        'patient_group', 'patient_count',
        'small_dice', 'small_dice_std', 'small_iou', 'small_iou_std',
        'small_hd', 'small_hd_std'
    ]]

    # Удаление групп без маленьких очагов
    small_lesion_metrics = (
        small_lesion_metrics
        [small_lesion_metrics['patient_group'].isin(
            ['multiple_only_small_lesion', 'multiple_mixed_lesion'])]
    )

    # Формирование датафрейма из метрики Dice по пациенту
    patients_dice_df = pd.DataFrame(list(patients_dice.items()), columns=['patient_id', 'dice'])

    return all_lesion_metrics, big_lesion_metrics, small_lesion_metrics, patients_dice_df



def best_worst_patient_segmentation(test_df, patients_dice_df):
    """Функция получения ID пациентов, для которых модель произвела лучшую и худшую сегментации"""

    # Исключение пациента без очага
    no_lesion_patient_id = test_df[test_df['lesion_label'] == 'no_lesion']['patient_id'].unique().item()
    patients_dice_df = patients_dice_df[patients_dice_df['patient_id'] != no_lesion_patient_id]

    # Выделение пациентов с лучшими и худшими метриками Dice
    best_dice_patients_idx = (
        patients_dice_df.sort_values('dice', ascending=False)
        .head(3)['patient_id'].tolist()
    )

    worst_dice_patients_idx = (
        patients_dice_df.sort_values('dice')
        .head(3)['patient_id'].tolist())

    return best_dice_patients_idx, worst_dice_patients_idx
