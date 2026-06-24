# dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as transforms


class AircraftDataset(Dataset):
    def __init__(self, data_dir='data', split='train',
                 image_size=224, augment=False):
        self.split = split
        self.data_dir = os.path.join(data_dir, split)
        self.image_size = image_size
        self.augment = augment

        # Проверяем существование папки
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Папка {self.data_dir} не найдена!")

        # Определяем трансформации
        self.transform = self._get_transforms()

        # Загружаем данные
        self.load_data()

    def _get_transforms(self):
        """Возвращает трансформации в зависимости от режима"""
        if self.split == 'train' and self.augment:
            # СИЛЬНЫЕ аугментации для маленького датасета
            return transforms.Compose([
                transforms.RandomResizedCrop(self.image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.05),  # иногда самолеты перевернуты
                transforms.RandomRotation(30),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.2, 0.2),
                    scale=(0.8, 1.2)
                ),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))  # случайное стирание
            ])
        else:
            # Для валидации/теста
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def load_data(self):
        """Загружает данные из папок"""
        try:
            self.classes = sorted([d for d in os.listdir(self.data_dir)
                                   if os.path.isdir(os.path.join(self.data_dir, d))])

            if not self.classes:
                raise ValueError(f"В папке {self.data_dir} не найдено классов!")

            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            print(f"Найдено классов в {self.split}: {len(self.classes)}")

            self.samples = []
            self.labels = []  # храним метки отдельно для балансировки

            for cls in self.classes:
                cls_path = os.path.join(self.data_dir, cls)

                # Проверяем существование папки класса
                if not os.path.exists(cls_path):
                    print(f"Предупреждение: папка класса {cls} не найдена!")
                    continue

                # Ищем изображения во всех подпапках (включая your_new_photos/)
                images = []
                for root, dirs, files in os.walk(cls_path):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            images.append(os.path.join(root, file))

                if not images:
                    print(f"Предупреждение: в классе {cls} нет изображений!")
                    continue

                for img_path in images:
                    self.samples.append({
                        'path': img_path,
                        'label': cls
                    })
                    self.labels.append(self.class_to_idx[cls])

            if not self.samples:
                raise ValueError(f"В {self.split} не найдено изображений!")

            print(f"Всего изображений в {self.split}: {len(self.samples)}")

            # Статистика по классам
            if self.labels:
                class_counts = np.bincount(self.labels)
                print("\nСтатистика по классам:")
                for idx, count in enumerate(class_counts):
                    print(f"  {self.classes[idx]}: {count} изображений")

                # Проверка дисбаланса
                max_count = np.max(class_counts)
                min_count = np.min(class_counts)
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                print(f"  Дисбаланс: {imbalance_ratio:.1f}x")

        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке данных: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Пробуем открыть изображение
                image = Image.open(sample['path']).convert('RGB')

                # Проверяем, что изображение загрузилось
                if image is None:
                    raise ValueError("Изображение None")

                # Применяем трансформации
                if self.transform:
                    image = self.transform(image)

                return image, self.class_to_idx[sample['label']]

            except Exception as e:
                if attempt == max_attempts - 1:
                    # Последняя попытка - возвращаем фиктивное изображение
                    print(f"⚠ Ошибка при загрузке {sample['path']}: {e}")
                    print(f"  Используется фиктивное изображение")

                    # Создаем фиктивное изображение
                    dummy_image = Image.new('RGB', (self.image_size, self.image_size), color='gray')

                    # Применяем базовые трансформации для фиктивного изображения
                    basic_transform = transforms.Compose([
                        transforms.Resize((self.image_size, self.image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    dummy_image = basic_transform(dummy_image)

                    return dummy_image, self.class_to_idx[sample['label']]


def create_dataloaders(data_dir='data', batch_size=32, num_workers=4,
                       image_size=224, balance_classes=True):
    """
    Создает DataLoader'ы для обучения и валидации с опциональной балансировкой

    Args:
        data_dir: Путь к данным
        batch_size: Размер батча
        num_workers: Количество рабочих процессов
        image_size: Размер изображения
        balance_classes: Балансировать ли классы (True/False)

    Returns:
        train_loader, val_loader, class_names
    """
    print("=" * 50)
    print("СОЗДАНИЕ DATALOADER'ОВ")
    print("=" * 50)

    # Проверяем существование папок
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(train_dir):
        raise ValueError(f"Папка train не найдена: {train_dir}")
    if not os.path.exists(val_dir):
        raise ValueError(f"Папка val не найдена: {val_dir}")

    # Создаем датасеты
    print("\nСоздание тренировочного датасета...")
    train_dataset = AircraftDataset(
        data_dir=data_dir,
        split='train',
        image_size=image_size,
        augment=True  # Аугментации только для обучения
    )

    print("\nСоздание валидационного датасета...")
    val_dataset = AircraftDataset(
        data_dir=data_dir,
        split='val',
        image_size=image_size,
        augment=False  # Без аугментаций для валидации
    )

    print(f"\nИтоговая статистика:")
    print(f"  Классы: {len(train_dataset.classes)}")
    print(f"  Тренировочные изображения: {len(train_dataset)}")
    print(f"  Валидационные изображения: {len(val_dataset)}")

    # Создаем DataLoader'ы
    if balance_classes and len(train_dataset.labels) > 0:
        print("\n🔧 Применение балансировки классов...")

        try:
            # Вычисляем веса для каждого класса
            class_counts = np.bincount(train_dataset.labels)
            print(f"  Количество по классам: {class_counts}")

            # Избегаем деления на ноль
            class_counts = np.maximum(class_counts, 1)
            class_weights = 1. / class_counts

            # Нормализуем веса
            class_weights = class_weights / class_weights.sum()

            # Веса для каждого сэмпла
            sample_weights = [class_weights[label] for label in train_dataset.labels]

            # Создаем сэмплер
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

            print(f"  Веса классов: {class_weights}")
            print(f"  Используется WeightedRandomSampler")

            # Создаем DataLoader с сэмплером
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,  # ЗАМЕНЯЕМ shuffle на sampler
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),  # Только для CUDA
                drop_last=True  # Игнорируем последний неполный батч
            )

        except Exception as e:
            print(f"  ⚠ Ошибка балансировки: {e}")
            print(f"  Используется обычный DataLoader")
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),  # Только для CUDA
                drop_last=True  # Игнорируем последний неполный батч
            )
    else:
        # Обычный DataLoader без балансировки
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),  # Только для CUDA
            drop_last=True  # Игнорируем последний неполный батч
        )

    # Валидационный DataLoader (без балансировки)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Только для CUDA
        drop_last=False  # Для валидации можно оставить неполные батчи
    )

    print(f"\n✓ DataLoader созданы:")
    print(f"  Train: {len(train_loader)} батчей по {batch_size}")
    print(f"  Val: {len(val_loader)} батчей по {batch_size}")

    return train_loader, val_loader, train_dataset.classes


def calculate_class_weights(dataset):
    """
    Вычисляет веса классов для WeightedRandomSampler

    Args:
        dataset: PyTorch Dataset с атрибутом labels

    Returns:
        sample_weights: список весов для каждого сэмпла
    """
    if not hasattr(dataset, 'labels'):
        raise ValueError("Dataset должен иметь атрибут 'labels'")

    # Считаем количество сэмплов в каждом классе
    class_counts = np.bincount(dataset.labels)

    # Избегаем деления на ноль
    class_counts = np.maximum(class_counts, 1)

    # Вычисляем веса (обратно пропорционально количеству сэмплов)
    class_weights = 1. / class_counts

    # Нормализуем веса
    class_weights = class_weights / class_weights.sum()

    # Создаем веса для каждого сэмпла
    sample_weights = [class_weights[label] for label in dataset.labels]

    return sample_weights


def test_dataset():
    """Тестирование датасета"""
    print("\n" + "=" * 50)
    print("ТЕСТИРОВАНИЕ DATASET")
    print("=" * 50)

    try:
        # Тест тренировочного датасета
        print("\n1. Тест тренировочного датасета:")
        train_dataset = AircraftDataset(split='train', augment=True)
        print(f"   ✓ Train dataset: {len(train_dataset)} images")
        print(f"   ✓ Classes: {len(train_dataset.classes)}")

        # Тест валидационного датасета
        print("\n2. Тест валидационного датасета:")
        val_dataset = AircraftDataset(split='val', augment=False)
        print(f"   ✓ Val dataset: {len(val_dataset)} images")

        # Проверяем первый элемент
        print("\n3. Проверка загрузки изображения:")
        img, label = train_dataset[0]
        print(f"   ✓ Размер изображения: {img.shape}")
        print(f"   ✓ Метка: {label} ({train_dataset.classes[label]})")

        # Тест create_dataloaders с балансировкой
        print("\n4. Тест create_dataloaders с балансировкой:")
        train_loader, val_loader, classes = create_dataloaders(
            batch_size=4,
            balance_classes=True
        )
        print(f"   ✓ DataLoader train: {len(train_loader)} батчей")
        print(f"   ✓ DataLoader val: {len(val_loader)} батчей")

        # Проверяем батч
        print("\n5. Проверка батча:")
        images, labels = next(iter(train_loader))
        print(f"   ✓ Размер батча: {images.shape}")
        print(f"   ✓ Метки в батче: {labels.tolist()}")

        return True

    except Exception as e:
        print(f"\n✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dataset()
    if success:
        print("\n" + "=" * 50)
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("❌ ТЕСТЫ НЕ ПРОЙДЕНЫ")
        print("=" * 50)