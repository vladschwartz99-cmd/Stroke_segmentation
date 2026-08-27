import nibabel
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting
from src.preprocessing import build_loader
from src.evaluate_model import get_prediction, get_ensemble_prediction



def visualize_all_patient_images(patient_df):
    """Функция визуализации МР-изображений и маски сегментации"""

    fig, ax = plt.subplots(2, 2, figsize=(18, 10))

    # Вывод всех снимков и маски в трех проекциях
    for axes, label in zip(ax.flatten(), patient_df['label'].unique()):
        plotting.plot_anat(patient_df[patient_df['label'] == label]['file_path'].item(),
                           display_mode='ortho', title=label, axes=axes
        )

    fig.suptitle('Пример маски и всех МР-изображений для одного пациента')
    plt.show()



def visualize_mri_with_mask(mri_df, mask_df):
    """Функция визуализации МР-изображений с наложением маски сегментации"""

    fig, ax = plt.subplots(len(mri_df), figsize=(18, 10))

    # Вывод всех снимков с наложенной маской в трех проекциях
    for axes, label in zip(ax.flatten(), mri_df['label'].unique()):
        plotting.plot_roi(
            mask_df['file_path'].item(),
            mri_df[mri_df['label'] == label]['file_path'].item(),
            display_mode='ortho', title=label, axes=axes, alpha=0.2
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



def visualize_mri_with_two_mask(model, test_df, patient_id, protocols_list, threshold, small_model=None):
    """Функция визуализации МР-изображений с наложением исходной и предсказанной масок сегментации"""

    # Формируем датафрейм для одного пациента
    patient_df = test_df[test_df['patient_id'] == patient_id]

    # Формируем лоадер
    patient_loader = build_loader(
        patient_df, protocols_list, augmentations=False,
        batch_size=1, shuffle=False, save_transforms_meta=True
    )

    batch = next(iter(patient_loader))

    # Получаем полное изображение и его афинную матрицу
    full_image = batch['image']
    affine = batch['image'].affine.cpu().numpy()

    # Получаем предсказание и переводим в бинарную маску
    if small_model:
        pred = get_ensemble_prediction(full_image, model, small_model)

    else:
        pred = get_prediction(full_image, model)

    pred_mask = (pred[0, 0] > threshold).cpu().numpy()

    # Получаем маску исходной сегментации
    true_mask = batch['mask'][0, 0].cpu().numpy()

    # Переводим маски в нужный для визуализации формат
    true_mask = nibabel.Nifti1Image(true_mask.astype(np.uint8), affine)
    pred_mask = nibabel.Nifti1Image(pred_mask.astype(np.uint8), affine)

    fig, ax = plt.subplots(len(protocols_list), figsize=(18, 10))

    # Вывод всех снимков с наложенной маской в трех проекциях
    for axes, (n, protocol) in zip(ax, enumerate(protocols_list)):

        # Переводим изображение одного протокола в нужный формат
        image = full_image[0, n].cpu().numpy()
        image = nibabel.Nifti1Image(image.astype(np.float32), affine)

        # Визуализируем изображение с исходной маской
        display = plotting.plot_roi(
            roi_img=true_mask, bg_img=image, display_mode='ortho',
            title=protocol, axes=axes, alpha=0.4
        )

        # Накладываем предсказанную маску
        display.add_overlay(pred_mask, cmap='Reds', transparency=0.7, vmin=0, vmax=1)

    fig.suptitle('Сравнение исходной (синяя) и предсказанной (красная) масок сегментации')
    plt.show()
