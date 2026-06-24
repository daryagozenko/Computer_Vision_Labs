# checkpoints.py
import torch
import os
import json
import shutil
from datetime import datetime


class CheckpointManager:
    """Менеджер для работы с контрольными точками"""

    def __init__(self, checkpoint_dir='checkpoints'):
        """
        Args:
            checkpoint_dir (str): Папка для сохранения чекпоинтов
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Создаем папку для лучших моделей
        self.best_models_dir = os.path.join(checkpoint_dir, 'best_models')
        os.makedirs(self.best_models_dir, exist_ok=True)

        # Файл с историей чекпоинтов
        self.history_file = os.path.join(checkpoint_dir, 'checkpoint_history.json')

        # Загружаем историю если существует
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {
                'checkpoints': [],
                'best_models': [],
                'training_sessions': []
            }

    def save_checkpoint(self, epoch, model, optimizer, scheduler=None,
                        train_loss=None, val_loss=None,
                        train_acc=None, val_acc=None,
                        model_name='model', is_best=False, additional_info=None):
        """
        Сохраняет контрольную точку
        Args:
            epoch (int): Номер эпохи
            model: Модель PyTorch
            optimizer: Оптимизатор
            scheduler: Планировщик LR (опционально)
            train_loss (float): Loss на обучении
            val_loss (float): Loss на валидации
            train_acc (float): Accuracy на обучении
            val_acc (float): Accuracy на валидации
            model_name (str): Имя модели
            is_best (bool): Лучшая ли это модель
            additional_info (dict): Дополнительная информация
        Returns:
            str: Путь к сохраненному файлу
        """
        # Создаем чекпоинт
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'additional_info': additional_info or {}
        }

        # Добавляем состояние scheduler если есть
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # Определяем имя файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if is_best:
            filename = f"best_{model_name}_{timestamp}.pth"
            save_dir = self.best_models_dir
        else:
            filename = f"checkpoint_{model_name}_epoch{epoch:03d}_{timestamp}.pth"
            save_dir = self.checkpoint_dir

        # Полный путь к файлу
        filepath = os.path.join(save_dir, filename)

        # Сохраняем чекпоинт
        torch.save(checkpoint, filepath)

        # Добавляем в историю
        checkpoint_info = {
            'filename': filename,
            'path': filepath,
            'epoch': epoch,
            'model_name': model_name,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'timestamp': checkpoint['timestamp'],
            'is_best': is_best
        }

        self.history['checkpoints'].append(checkpoint_info)

        if is_best:
            self.history['best_models'].append(checkpoint_info)

        # Сохраняем обновленную историю
        self._save_history()

        print(f"Чекпоинт сохранен: {filepath}")
        if is_best:
            print(f"  ⭐ ЛУЧШАЯ МОДЕЛЬ! Accuracy: {val_acc:.2f}%")

        return filepath

    def load_checkpoint(self, filepath, model, optimizer=None, scheduler=None):
        """
        Загружает контрольную точку
        Args:
            filepath (str): Путь к файлу чекпоинта
            model: Модель PyTorch
            optimizer: Оптимизатор (опционально)
            scheduler: Планировщик LR (опционально)
        Returns:
            dict: Информация из чекпоинта
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Чекпоинт не найден: {filepath}")

        print(f"Загрузка чекпоинта: {filepath}")

        # Загружаем чекпоинт
        checkpoint = torch.load(filepath, map_location='cpu')

        # Загружаем веса модели
        model.load_state_dict(checkpoint['model_state_dict'])

        # Загружаем состояние оптимизатора если передан
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Загружаем состояние scheduler если передан
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Выводим информацию о чекпоинте
        print(f"  Эпоха: {checkpoint['epoch']}")
        print(f"  Модель: {checkpoint.get('model_name', 'Unknown')}")

        if checkpoint.get('train_acc') is not None:
            print(f"  Train Accuracy: {checkpoint['train_acc']:.2f}%")
        if checkpoint.get('val_acc') is not None:
            print(f"  Val Accuracy: {checkpoint['val_acc']:.2f}%")

        print(f"  Дата: {checkpoint.get('timestamp', 'Unknown')}")

        return checkpoint

    def get_latest_checkpoint(self, model_name=None):
        """
        Возвращает последний чекпоинт
        Args:
            model_name (str): Фильтр по имени модели
        Returns:
            str: Путь к последнему чекпоинту или None
        """
        checkpoints = self.get_all_checkpoints(model_name)
        if checkpoints:
            return checkpoints[-1]['path']
        return None

    def get_best_checkpoint(self, model_name=None):
        """
        Возвращает лучший чекпоинт по accuracy
        Args:
            model_name (str): Фильтр по имени модели
        Returns:
            str: Путь к лучшему чекпоинту или None
        """
        best_checkpoints = self.history['best_models']

        if model_name:
            best_checkpoints = [c for c in best_checkpoints if c['model_name'] == model_name]

        if not best_checkpoints:
            return None

        # Находим чекпоинт с максимальной точностью
        best = max(best_checkpoints, key=lambda x: x['val_acc'] if x['val_acc'] is not None else 0)
        return best['path']

    def get_all_checkpoints(self, model_name=None):
        """
        Возвращает все чекпоинты
        Args:
            model_name (str): Фильтр по имени модели
        Returns:
            list: Список информации о чекпоинтах
        """
        checkpoints = self.history['checkpoints']

        if model_name:
            checkpoints = [c for c in checkpoints if c['model_name'] == model_name]

        return checkpoints

    def delete_checkpoint(self, filepath):
        """
        Удаляет чекпоинт
        Args:
            filepath (str): Путь к файлу чекпоинта
        """
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Чекпоинт удален: {filepath}")

            # Удаляем из истории
            self.history['checkpoints'] = [
                c for c in self.history['checkpoints'] if c['path'] != filepath
            ]
            self.history['best_models'] = [
                c for c in self.history['best_models'] if c['path'] != filepath
            ]

            self._save_history()

    def cleanup_old_checkpoints(self, keep_last_n=5):
        """
        Удаляет старые чекпоинты, оставляя только последние N
        Args:
            keep_last_n (int): Сколько чекпоинтов оставить
        """
        checkpoints = self.get_all_checkpoints()

        if len(checkpoints) <= keep_last_n:
            print(f"Всего {len(checkpoints)} чекпоинтов, удаление не требуется")
            return

        # Сортируем по времени (старые сначала)
        checkpoints.sort(key=lambda x: x.get('timestamp', ''))

        # Удаляем старые
        to_delete = checkpoints[:-keep_last_n]

        for checkpoint in to_delete:
            if os.path.exists(checkpoint['path']):
                os.remove(checkpoint['path'])
                print(f"Удален старый чекпоинт: {checkpoint['filename']}")

        # Обновляем историю
        self.history['checkpoints'] = checkpoints[-keep_last_n:]
        self._save_history()

    def _save_history(self):
        """Сохраняет историю в JSON файл"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def print_summary(self):
        """Печатает сводку по чекпоинтам"""
        print("\n" + "=" * 60)
        print("СВОДКА ПО ЧЕКПОИНТАМ")
        print("=" * 60)

        all_checkpoints = self.get_all_checkpoints()
        best_checkpoints = self.history['best_models']

        print(f"Всего чекпоинтов: {len(all_checkpoints)}")
        print(f"Лучших моделей: {len(best_checkpoints)}")

        if best_checkpoints:
            print("\nЛучшие модели:")
            for cp in sorted(best_checkpoints, key=lambda x: x['val_acc'] or 0, reverse=True)[:3]:
                print(f"  {cp['filename']}:")
                print(f"    Accuracy: {cp['val_acc']:.2f}%")
                print(f"    Эпоха: {cp['epoch']}")
                print(f"    Дата: {cp['timestamp'][:10]}")

        # Группируем по имени модели
        model_groups = {}
        for cp in all_checkpoints:
            model_name = cp['model_name']
            if model_name not in model_groups:
                model_groups[model_name] = []
            model_groups[model_name].append(cp)

        print("\nПо моделям:")
        for model_name, cps in model_groups.items():
            accuracies = [cp['val_acc'] for cp in cps if cp['val_acc'] is not None]
            if accuracies:
                avg_acc = sum(accuracies) / len(accuracies)
                best_acc = max(accuracies)
                print(f"  {model_name}: {len(cps)} чекпоинтов, "
                      f"средняя accuracy: {avg_acc:.2f}%, "
                      f"лучшая: {best_acc:.2f}%")

        print("=" * 60)


if __name__ == "__main__":
    # Пример использования
    manager = CheckpointManager()

    # Печатаем сводку
    manager.print_summary()

    # Пример: получение лучшего чекпоинта
    best_checkpoint = manager.get_best_checkpoint()
    if best_checkpoint:
        print(f"\nЛучший чекпоинт: {best_checkpoint}")
    else:
        print("\nЧекпоинты не найдены")