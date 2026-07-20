import torch
import tqdm
from pathlib import Path
from collections import defaultdict
from IPython.display import clear_output
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference


# Устройство для вычислений
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def soft_dice_coef(pred, target, smooth=1e-5):
    """Функция, рассчитывающая коэффициент Дайса для предсказанных вероятностей и исходных масок"""

    # Преобразование предсказания модели в вероятности
    pred = torch.sigmoid(pred.float())

    # Вычисление пересечения
    intersection = (pred * target).sum(dim=(2, 3, 4))

    # Вычисление объединения
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))

    # Вычисление коэффициента Dice
    dice = (2. * intersection + smooth) / (union + smooth)

    # Возврат значения функции потерь Dice
    return dice.mean()



def train_one_epoch(model, optimizer, criterion, scaler, train_loader):
    """Функция обучения модели в течении одной эпохи"""

    model.train()

    # Переменные для подсчета метрик
    train_loss = 0.0
    train_dice = 0.0

    for images, masks in tqdm.tqdm_notebook(train_loader):

        # Перевод данных на устройство
        images, masks = images.to(device), masks.to(device)

        # Обнуление градиента
        optimizer.zero_grad()

        # Переход с float32 на float16
        with torch.autocast(device_type='cuda', dtype=torch.float16):

            # Получение ответа модели
            logits = model(images)

            # Извлечение ответа U-Net ++
            if isinstance(logits, list):
                logits = logits[-1]

            # Подсчет лосса
            loss = criterion(logits, masks)

        # Расчет градиента и шаг оптимизатора
        scaler.scale(loss).backward()
        scaler.step(optimizer)

        # Обновление коэффициента масштабирования
        scaler.update()

        # Расчет метрик по батчам
        train_loss += loss.item()
        train_dice += soft_dice_coef(logits, masks, smooth=1e-5).item()

    # Расчет метрик для всего train
    epoch_train_loss = train_loss / len(train_loader)
    epoch_train_dice = train_dice / len(train_loader)

    return epoch_train_loss, epoch_train_dice



def validate_one_epoch(model, criterion, val_loader):
    """Функция валидации модели в течении одной эпохи"""

    model.eval()

    val_loss = 0.0
    val_dice = 0.0

    with torch.no_grad():

        for images, masks in tqdm.tqdm_notebook(val_loader):

            # Перевод данных на устройство
            images, masks = images.to(device), masks.to(device)

            # Переход с float32 на float16
            with torch.autocast(device_type='cuda', dtype=torch.float16):

                # Патчинг изображения и получение ответа модели
                val_outputs = sliding_window_inference(
                    inputs=images, roi_size=(96, 96, 96),
                    sw_batch_size=1, predictor=model,
                    overlap=0.5, mode='gaussian'
                )

                # Опять же для U-Net ++
                if isinstance(val_outputs, list):
                    val_outputs = val_outputs[-1]

                # Подсчет лосса
                loss = criterion(val_outputs, masks)

            # Расчет метрик по батчам
            val_loss += loss.item()
            val_dice += soft_dice_coef(val_outputs, masks, smooth=1e-5).item()

    # Расчет метрик для всего val
    epoch_val_loss = val_loss / len(val_loader)
    epoch_val_dice = val_dice / len(val_loader)

    return epoch_val_loss, epoch_val_dice



def plot_learning_curves(history):
    """Функция для вывода графиков лосса и коэффициента Дайса во время обучения"""

    fig = plt.figure(figsize=(20, 7))

    plt.subplot(1,2,1)
    plt.title('Loss', fontsize=15)
    plt.plot(history['loss']['train'], label='train')
    plt.plot(history['loss']['val'], label='val')
    plt.ylabel('loss', fontsize=15)
    plt.xlabel('epoch_number', fontsize=15)
    plt.legend()

    plt.subplot(1,2,2)
    plt.title('Dice', fontsize=15)
    plt.plot(history['dice']['train'], label='train')
    plt.plot(history['dice']['val'], label='val')
    plt.ylabel('dice', fontsize=15)
    plt.xlabel('epoch_number', fontsize=15)
    plt.legend()
    plt.show()



def train_model(
        model, criterion, optimizer, scheduler,
        train_loader, val_loader, weights_dir_name,
        n_epochs=200, patience = 10,
):
    """Функция полного обучения и валидации модели в течении n эпох,
            с ранней остановкой и сохранением лучших весов"""

    model = model.to(device)

    # Словарь, для сохранения метрик между эпохами
    history = defaultdict(lambda: defaultdict(list))

    # Переменные для преждевременной остановки
    best_val_dice = 0.0
    epochs_without_improvement = 0

    # Создание папки для сохранения весов
    weights_dir = Path('weights') / weights_dir_name
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Инициализация скейлера
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(n_epochs):

        # Обучение
        train_loss, train_dice = train_one_epoch(model, optimizer, criterion, scaler, train_loader)

        # Сохранение train метрик эпохи
        history['loss']['train'].append(train_loss)
        history['dice']['train'].append(train_dice)

        # Валидация
        val_loss, val_dice = validate_one_epoch(model, criterion, val_loader)

        # Сохранение val метрик эпохи
        history['loss']['val'].append(val_loss)
        history['dice']['val'].append(val_dice)

        # Шаг планировщика
        scheduler.step(val_dice)

        if val_dice > best_val_dice:

            # Пересохранение лучшей метрики
            best_val_dice = val_dice

            # Сохранение весов модели
            save_path = weights_dir / f'best_{model.__class__.__name__}.pth'
            torch.save(model.state_dict(), save_path)

            # Обнуление счетчика эпох без улучшения
            epochs_without_improvement = 0

        # Шаг счетчика при отсутствии улучшения
        else:
            epochs_without_improvement += 1

        # Ранняя остановка
        if epochs_without_improvement >= patience:
            break

        # Избегание захломления вывода функции
        clear_output()

        # Вывод метрик эпохи
        print(f'Epoch {epoch + 1} of {n_epochs}')
        print(f'train loss: \t{train_loss:.6f}')
        print(f'val loss: \t{val_loss:.6f}')
        print(f'train dice: \t\t\t{train_dice:.2f}')
        print(f'val dice: \t\t\t{val_dice:.2f}')

        # Вывод кривых обучения по сохраненным значениям метрик
        plot_learning_curves(history)
