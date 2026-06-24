# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import time
import os
import sys

# Исправляем импорт (у вас опечатка: chekpoints -> checkpoints)
try:
    from checkpoints import CheckpointManager
except ImportError:
    # Попробуем с опечаткой
    from chekpoints import CheckpointManager

# Добавляем путь к текущей директории для импорта наших модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import create_dataloaders
from models import create_model
from utils import plot_training_history, EarlyStopping


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Одна эпоха обучения"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Обнуляем градиенты
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Статистика
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Прогресс каждые 10 батчей
        if (batch_idx + 1) % 10 == 0:
            print(f'  Batch {batch_idx + 1}/{len(dataloader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100. * correct / total:.2f}%', end='\r')

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Одна эпоха валидации"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def train_model(config):
    """
    Основная функция обучения
    Args:
        config (dict): Конфигурация обучения
    """
    print("\n" + "=" * 60)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 60)

    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    # Создаем DataLoader'ы - ИСПРАВЛЕНО: убран use_fgvc
    print("\nЗагрузка данных...")
    try:
        train_loader, val_loader, class_names = create_dataloaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            image_size=224,
            balance_classes=config.get('balance_classes', True)  # добавляем балансировку
        )
    except Exception as e:
        print(f"❌ Ошибка при загрузке данных: {e}")
        print("\nВозможные причины:")
        print("1. Структура папок неправильная")
        print("2. Нет папок train/ или val/")
        print("3. В папках нет классов")
        print("\nПроверьте структуру данных:")
        print("  data/train/класс1/изображения.jpg")
        print("  data/train/класс2/изображения.jpg")
        print("  data/val/класс1/изображения.jpg")
        print("  data/val/класс2/изображения.jpg")
        return None, None, None

    print(f"Классы: {class_names}")
    print(f"Размер обучающей выборки: {len(train_loader.dataset)}")
    print(f"Размер валидационной выборки: {len(val_loader.dataset)}")

    # Создаем модель
    print("\nСоздание модели...")
    model, model_info = create_model(
        model_name=config['model_name'],
        num_classes=len(class_names),
        pretrained=config['pretrained'],
        fine_tune_mode=config['fine_tune_mode'],  # ← новый параметр
        device=device
    )

    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Планировщик обучения
    if config['scheduler'] == 'step':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    elif config['scheduler'] == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    else:
        scheduler = None

    # Ранняя остановка
    early_stopping = EarlyStopping(patience=config['patience'], verbose=True)

    # Менеджер чекпоинтов
    checkpoint_manager = CheckpointManager(config['checkpoint_dir'])

    # История обучения
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'learning_rate': []
    }

    print(f"\nОбучение на {config['num_epochs']} эпох...")
    print("=" * 60)

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(config['num_epochs']):
        epoch_start = time.time()

        print(f"\nЭпоха {epoch + 1}/{config['num_epochs']}")
        print("-" * 40)

        # Обучение
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Валидация
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )

        # Обновление планировщика
        if scheduler is not None:
            if config['scheduler'] == 'plateau':
                scheduler.step(val_acc)
            else:
                scheduler.step()

        # Текущий learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Сохраняем историю
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(current_lr)

        # Время эпохи
        epoch_time = time.time() - epoch_start

        # Вывод статистики
        print(f"\nРезультаты:")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Время эпохи: {epoch_time:.1f} секунд")

        # Сохраняем обычный чекпоинт
        checkpoint_manager.save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            model_name=config['model_name'],
            is_best=False,
            additional_info={
                'classes': class_names,
                'config': config,
                'model_info': model_info
            }
        )

        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_manager.save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                model_name=config['model_name'],
                is_best=True,
                additional_info={
                    'classes': class_names,
                    'config': config,
                    'model_info': model_info
                }
            )

        # Проверка ранней остановки
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"\nРанняя остановка на эпохе {epoch + 1}")
            break

    # Общее время обучения
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 60)
    print(f"Общее время: {total_time:.1f} секунд")
    print(f"Лучшая точность: {best_val_acc:.2f}%")
    print(f"Финальная точность: {history['val_acc'][-1]:.2f}%")

    # Визуализация истории обучения
    try:
        plot_training_history(history, save_path='training_history.png')
        print("График обучения сохранен: training_history.png")
    except Exception as e:
        print(f"Не удалось сохранить график: {e}")

    # Сохраняем финальную модель
    final_checkpoint = checkpoint_manager.save_checkpoint(
        epoch=config['num_epochs'] - 1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loss=history['train_loss'][-1],
        val_loss=history['val_loss'][-1],
        train_acc=history['train_acc'][-1],
        val_acc=history['val_acc'][-1],
        model_name=config['model_name'],
        is_best=False,
        additional_info={
            'classes': class_names,
            'config': config,
            'model_info': model_info,
            'total_time': total_time,
            'best_val_acc': best_val_acc
        }
    )

    print(f"\nФинальная модель сохранена: {final_checkpoint}")

    # Печатаем сводку по чекпоинтам
    checkpoint_manager.print_summary()

    return model, history, checkpoint_manager


def main():
    """Основная функция для запуска обучения"""

    # Конфигурация обучения
    config = {
        # Данные
        'data_dir': 'data',
        'batch_size': 16,
        'num_workers': 2,

        # Модель
        'model_name': 'resnet50',
        'pretrained': True,
        'fine_tune_mode': 'all',

        # Обучение
        'num_epochs': 20,
        'learning_rate': 0.0003,
        'scheduler': 'plateau',
        'patience': 5,

        # Балансировка
        'balance_classes': True,
        'image_size': 224,

        # Чекпоинты
        'checkpoint_dir': 'checkpoints',
    }

    # Запуск обучения
    try:
        model, history, checkpoint_manager = train_model(config)
        if model is not None:
            print("\n✅ Обучение успешно завершено!")
        else:
            print("\n❌ Обучение не удалось запустить")
    except Exception as e:
        print(f"\n❌ Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()