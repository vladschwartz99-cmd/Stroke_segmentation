import torch
import numpy as np
import pandas as pd
from monai.inferers import sliding_window_inference
from scipy.ndimage import label
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

    # 3. Вычисление расстояния Хаусдорфа (безопасно, обе маски не пустые)
    hd = hd95(pred_mask, true_mask)

    return dice, iou, hd



def get_overlaps(pred_mask, true_mask):

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


    for pred_mask , true_mask in pred_true_masks:

        # Преобразование предсказанной маски в бинарную на основе порога уверенности
        pred_mask = pred_mask > threshold

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

    # Формирование словаря метрик
    metrics_dict = {
        'threshold': threshold,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'dice': np.mean(dice_list) if dice_list else 0.0,
        'iou': np.mean(iou_list) if iou_list else 0.0,
        'hd': np.mean(valid_hd) if valid_hd else 0.0,
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
            'threshold', 'recall', 'precision',
            'f1', 'dice', 'iou', 'hd'
        ]
    )

    # Находим строку максимального значения Dice
    thresholds_metrics = thresholds_metrics.loc[thresholds_metrics['dice'].idxmax()]

    return thresholds_metrics.to_frame().T



def groups_metrics_reports(model, patients_df, protocols_list, threshold=0.5):
    """Функция, рассчитывающая метрики сегментации по группам пациентов и по объему очагов"""

    # Список для сохранения итоговых метрик по группам
    groups_metrics = []

    # Получение названий групп пациентов
    patients_groups = patients_df['lesion_label'].unique()

    # Для каждой группы
    for group in patients_groups:

        # Формирование подвыборки из пациентов определенной группы
        group_df = patients_df[patients_df['lesion_label'] == group]

        n_patients = len(group_df['patient_id'].unique())

        # Преобразование в лоадер
        group_loader = build_loader(
            group_df, protocols_list, augmentations=False,
            batch_size=1, shuffle=False
        )

        # Для каждого изображения
        for image, true_mask in group_loader:

            # Получение предсказанной маски
            pred_mask = get_prediction(image, model)

            # Преобразование в бинарную маску по порогу
            pred_mask = pred_mask > threshold

            # Перевод масок на cpu
            pred_mask, true_mask = pred_mask.to('cpu'), true_mask.to('cpu')

            # Подсчет количества очагов
            pred_labels, pred_n_lesions = label(pred_mask)
            true_labels, true_n_lesions = label(true_mask)

            # Метрики по очагам для подсчета
            all_lesion_count = 0
            big_lesion_count = 0
            small_lesion_count = 0

            all_tp, big_tp, small_tp = 0, 0, 0
            all_fp, big_fp, small_fp = 0, 0, 0
            all_fn, big_fn, small_fn = 0, 0, 0

            all_dice, big_dice, small_dice = [], [], []
            all_iou, big_iou, small_iou = [], [], []
            all_hd, big_hd, small_hd = [], [], []

            # Список для сохранения метрик по пациентам
            metrics_for_patient = []

            # Расчет метрик для каждого очага исходной разметки
            for true_id in range(1, true_n_lesions + 1):
                true_lesion_mask = (true_labels == true_id)

                # Фильтрация мелких очагов-артефактов
                if np.sum(true_lesion_mask) < 25:
                    continue

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

                    # Расчет доли пересечения истинной и предсказанной масок
                    pred_overlap, true_overlap = get_overlaps(pred_lesion_mask, true_lesion_mask)

                    # Обновление счетчиков и списков
                    all_lesion_count += 1
                    all_dice.append(dice)
                    all_iou.append(iou)
                    all_hd.append(hd)

                    # Если маски пересекаются (с небольшим порогом из-за большого количества мелких очагов)
                    if true_overlap >= 0.15 and pred_overlap >= 0.15:
                        all_tp += 1

                    # Если маски не пересекаются / слабо пересекаются
                    else:
                        all_fn += 1

                        # Если предсказанная маска не пуста
                        if np.any(pred_mask):
                            all_fp += 1

                    # Обновление счетчиков и списков для крупных очагов
                    if np.sum(true_lesion_mask) > 30000:

                        big_lesion_count += 1
                        big_dice.append(dice)
                        big_iou.append(iou)
                        big_hd.append(hd)

                        # Если маски пересекаются (с небольшим порогом из-за большого количества мелких очагов)
                        if true_overlap >= 0.15 and pred_overlap >= 0.15:
                            big_tp += 1

                        # Если маски не пересекаются / слабо пересекаются
                        else:
                            big_fn += 1

                            # Если предсказанная маска не пуста
                            if np.any(pred_mask):
                                big_fp += 1

                    # Обновление счетчиков и списков для мелких очагов
                    else:

                        small_lesion_count += 1
                        small_dice.append(dice)
                        small_iou.append(iou)
                        small_hd.append(hd)

                        # Если маски пересекаются (с небольшим порогом из-за большого количества мелких очагов)
                        if true_overlap >= 0.15 and pred_overlap >= 0.15:
                            small_tp += 1

                        # Если маски не пересекаются / слабо пересекаются
                        else:
                            small_fn += 1

                            # Если предсказанная маска не пуста
                            if np.any(pred_mask):
                                small_fp += 1

                # Если модель не нашла истинный очаг
                else:

                    # Обновление счетчиков
                    all_lesion_count += 1
                    all_fn += 1
                    all_dice.append(0)
                    all_iou.append(0)

                    # Обновление счетчиков для крупных очагов
                    if np.sum(true_lesion_mask) > 30000:

                        big_lesion_count += 1
                        big_fn += 1
                        big_dice.append(0)
                        big_iou.append(0)

                    # Обновление счетчиков для мелких очагов
                    else:

                        small_lesion_count += 1
                        small_fn += 1
                        small_dice.append(0)
                        small_iou.append(0)

            # Поиск ложных предсказаний модели
            for pred_id in range(1, pred_n_lesions + 1):
                pred_lesion_mask = (pred_labels == pred_id)

                # Проверяем, какие ID предсказаний пересекаются с текущим реальным очагом
                overlapping_true_ids = np.unique(true_labels[pred_lesion_mask])
                overlapping_true_ids = overlapping_true_ids[overlapping_true_ids > 0]

                # Пропуск совпадений
                if len(overlapping_true_ids) > 0:
                    continue

                # В случае ложного очага
                else:

                    # Обновление счетчиков
                    all_fp += 1

                    # Обновление счетчиков для крупных очагов
                    if np.sum(pred_lesion_mask) > 30000:

                        big_fp += 1

                    # Обновление счетчиков для мелких очагов
                    else:

                        small_fp += 1

            # Вычисление recall для всех видов очагов
            all_recall = all_tp / (all_tp + all_fn) if all_tp + all_fn > 0 else 0.0
            big_recall = big_tp / (big_tp + big_fn) if big_tp + big_fn > 0 else 0.0
            small_recall = small_tp / (small_tp + small_fn) if small_tp + small_fn > 0 else 0.0

            # Вычисление precision для всех видов очагов
            all_precision = all_tp / (all_tp + all_fp) if all_tp + all_fp > 0 else 0.0
            big_precision = big_tp / (big_tp + big_fp) if big_tp + big_fp > 0 else 0.0
            small_precision = small_tp / (small_tp + small_fp) if small_tp + small_fp > 0 else 0.0

            # Вычисление f1 для всех видов очагов
            all_f1 = (
                2 * (all_precision * all_recall) /
                (all_precision + all_recall)
                if all_precision + all_recall > 0 else 0.0
            )
            big_f1 = (
                2 * (big_precision * big_recall) /
                (big_precision + big_recall)
                if big_precision + big_recall > 0 else 0.0
            )
            small_f1 = (
                2 * (small_precision * small_recall) /
                (small_precision + small_recall)
                if small_precision + small_recall > 0 else 0.0
            )

            # Удаление Nan из списка HD
            valid_all_hd = [x for x in all_hd if not np.isnan(x)]
            valid_big_hd = [x for x in big_hd if not np.isnan(x)]
            valid_small_hd = [x for x in small_hd if not np.isnan(x)]

            # Добавление метрик в список по пациентам
            metrics_for_patient.append({

                'patient_group': group,

                'all_recall': all_recall,
                'big_recall': big_recall,
                'small_recall': small_recall,

                'all_precision': all_precision,
                'big_precision': big_precision,
                'small_precision': small_precision,

                'all_f1': all_f1,
                'big_f1': big_f1,
                'small_f1': small_f1,

                'all_dice': np.mean(all_dice) if all_dice else 0.0,
                'big_dice': np.mean(big_dice) if big_dice else 0.0,
                'small_dice': np.mean(small_dice) if small_dice else 0.0,

                'all_iou': np.mean(all_iou) if all_iou else 0.0,
                'big_iou': np.mean(big_iou) if big_iou else 0.0,
                'small_iou': np.mean(small_iou) if small_iou else 0.0,

                'all_hd': np.mean(valid_all_hd) if all_hd else 0.0,
                'big_hd': np.mean(valid_big_hd) if big_hd else 0.0,
                'small_hd': np.mean(valid_small_hd) if small_hd else 0.0,
            })

        # Преобразование метрик пациентов группы в датафрейм
        group_metrics_df = pd.DataFrame(
            metrics_for_patient,
            columns=[
                'patient_group',
                'all_recall', 'big_recall', 'small_recall',
                'all_precision', 'big_precision', 'small_precision',
                'all_f1', 'big_f1', 'small_f1',
                'all_dice', 'big_dice', 'small_dice',
                'all_iou', 'big_iou', 'small_iou',
                'all_hd', 'big_hd', 'small_hd'
            ]
        )

        # Добавление средних метрик в общий список
        groups_metrics.append({

            'patient_group': group,
            'patient_count': n_patients,

            'all_recall': group_metrics_df['all_recall'].mean(),
            'big_recall': group_metrics_df['big_recall'].mean(),
            'small_recall': group_metrics_df['small_recall'].mean(),

            'all_precision': group_metrics_df['all_precision'].mean(),
            'big_precision': group_metrics_df['big_precision'].mean(),
            'small_precision': group_metrics_df['small_precision'].mean(),

            'all_f1': group_metrics_df['all_f1'].mean(),
            'big_f1': group_metrics_df['big_f1'].mean(),
            'small_f1': group_metrics_df['small_f1'].mean(),

            'all_dice': group_metrics_df['all_dice'].mean(),
            'big_dice': group_metrics_df['big_dice'].mean(),
            'small_dice': group_metrics_df['small_dice'].mean(),

            'all_iou': group_metrics_df['all_iou'].mean(),
            'big_iou': group_metrics_df['big_iou'].mean(),
            'small_iou': group_metrics_df['small_iou'].mean(),

            'all_hd': group_metrics_df['all_hd'].mean(),
            'big_hd': group_metrics_df['big_hd'].mean(),
            'small_hd': group_metrics_df['small_hd'].mean(),
        })

    # Преобразование списка средних метрик в датафрейм
    full_metrics_df = pd.DataFrame(
        groups_metrics,
        columns=[
            'patient_group', 'patient_count',
            'all_recall', 'big_recall', 'small_recall',
            'all_precision', 'big_precision', 'small_precision',
            'all_f1', 'big_f1', 'small_f1',
            'all_dice', 'big_dice', 'small_dice',
            'all_iou', 'big_iou', 'small_iou',
            'all_hd', 'big_hd', 'small_hd'
        ]
    )

    # Выделение метрик для очагов всех объемов
    all_lesion_metrics = full_metrics_df[[
        'patient_group', 'patient_count', 'all_recall',
        'all_precision', 'all_f1', 'all_dice', 'all_iou', 'all_hd'
    ]]

    # Выделение метрик для очагов больших объемов
    big_lesion_metrics = full_metrics_df[[
        'patient_group', 'patient_count', 'big_recall',
        'big_precision', 'big_f1', 'big_dice', 'big_iou', 'big_hd'
    ]]

    # Удаление групп без крупных очагов
    big_lesion_metrics = (
        big_lesion_metrics
        [big_lesion_metrics['patient_group'].isin(
        ['one_large_lesion', 'multiple_mixed_lesion'])]
    )

    # Выделение метрик для очагов мелких объемов
    small_lesion_metrics = full_metrics_df[[
        'patient_group', 'patient_count', 'small_recall',
        'small_precision', 'small_f1', 'small_dice', 'small_iou', 'small_hd'
    ]]

    # Удаление групп без маленьких очагов
    small_lesion_metrics = (
        small_lesion_metrics
        [small_lesion_metrics['patient_group'].isin(
        ['multiple_only_small_lesion', 'one_small_lesion', 'multiple_mixed_lesion'])]
    )

    return all_lesion_metrics, big_lesion_metrics, small_lesion_metrics
