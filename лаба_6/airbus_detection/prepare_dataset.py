# prepare_dataset.py
import os
import shutil
import pandas as pd
from tqdm import tqdm
import argparse


def prepare_fgvc_dataset(dataset_path, output_dir='data', num_classes=20):
    """
    Подготавливает FGVC Aircraft Dataset для обучения
    Args:
        dataset_path: Путь к распакованному датасету (fgvc-aircraft-2013b)
        output_dir: Папка для выходных данных
        num_classes: Сколько классов использовать (первые N)
    """
    print("Подготовка FGVC Aircraft Dataset...")

    # Пути к файлам
    images_dir = os.path.join(dataset_path, 'data', 'images')
    train_csv = os.path.join(dataset_path, 'train.csv')
    val_csv = os.path.join(dataset_path, 'val.csv')
    test_csv = os.path.join(dataset_path, 'test.csv')

    # Проверяем существование файлов
    if not os.path.exists(images_dir):
        print(f"Ошибка: папка {images_dir} не найдена!")
        return

    # Создаем структуру папок
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    # Функция для обработки CSV файла
    def process_csv(csv_file, split_name, limit_classes=None):
        if not os.path.exists(csv_file):
            print(f"Внимание: файл {csv_file} не найден")
            return {}

        print(f"\nОбработка {split_name}...")
        df = pd.read_csv(csv_file, header=None)

        # Создаем словарь: класс -> список изображений
        class_dict = {}

        for _, row in tqdm(df.iterrows(), total=len(df)):
            if len(row) >= 2:
                image_name = str(row[0]).strip()
                class_name = str(row[1]).strip()

                # Если указано ограничение по классам
                if limit_classes and class_name not in limit_classes:
                    continue

                if class_name not in class_dict:
                    class_dict[class_name] = []

                class_dict[class_name].append(image_name)

        return class_dict

    # Получаем все классы из train и val
    print("Определение классов...")
    train_classes = process_csv(train_csv, 'train')
    val_classes = process_csv(val_csv, 'val')
    test_classes = process_csv(test_csv, 'test')

    # Объединяем все классы
    all_classes = list(set(list(train_classes.keys()) + list(val_classes.keys()) + list(test_classes.keys())))

    # Сортируем и ограничиваем количество классов (если нужно)
    all_classes.sort()
    if num_classes and num_classes < len(all_classes):
        selected_classes = all_classes[:num_classes]
        print(f"Используем первые {num_classes} классов из {len(all_classes)}")
    else:
        selected_classes = all_classes
        print(f"Используем все {len(all_classes)} классов")

    # Выводим список выбранных классов
    print("\nВыбранные классы:")
    for i, cls in enumerate(selected_classes[:20]):  # Показываем первые 20
        print(f"  {i + 1:3d}. {cls}")
    if len(selected_classes) > 20:
        print(f"  ... и еще {len(selected_classes) - 20} классов")

    # Копируем изображения для каждого сплита
    total_copied = 0

    for split_name, class_dict in [('train', train_classes), ('val', val_classes)]:
        print(f"\nКопирование изображений для {split_name}...")

        for class_name in selected_classes:
            if class_name not in class_dict:
                continue

            # Создаем папку для класса
            class_output_dir = os.path.join(output_dir, split_name, class_name)
            os.makedirs(class_output_dir, exist_ok=True)

            # Копируем изображения
            for img_name in class_dict[class_name]:
                # Функция для поиска реального файла
                def find_real_file(base_name):
                    """Ищет реальный файл (не симлинк)"""
                    # Пробуем различные расширения
                    for ext in ['', '.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        potential_path = os.path.join(images_dir, f"{base_name}{ext}")

                        if os.path.exists(potential_path):
                            # Если это симлинк, получаем реальный путь
                            if os.path.islink(potential_path):
                                real_path = os.path.realpath(potential_path)
                                if os.path.exists(real_path):
                                    return real_path
                            else:
                                return potential_path

                    # Если не нашли с расширением, ищем по части имени
                    for root, dirs, files in os.walk(images_dir):
                        for file in files:
                            if file.startswith(base_name) or base_name in file:
                                full_path = os.path.join(root, file)
                                # Проверяем что это файл, а не симлинк
                                if os.path.isfile(full_path) and not os.path.islink(full_path):
                                    return full_path

                    return None

                # Ищем реальный файл
                base_name = img_name.split('.')[0] if '.' in img_name else img_name
                src_path = find_real_file(base_name)

                if src_path and os.path.exists(src_path):
                    # Используем оригинальное имя файла для сохранения
                    dst_filename = os.path.basename(src_path)
                    dst_path = os.path.join(class_output_dir, dst_filename)

                    try:
                        shutil.copy2(src_path, dst_path)
                        total_copied += 1
                        if total_copied % 100 == 0:
                            print(f"  Скопировано: {total_copied}...")
                    except Exception as e:
                        print(f"  Ошибка копирования {src_path}: {e}")
                else:
                    print(f"  ⚠ Файл не найден: {img_name} (базовое имя: {base_name})")

    print(f"\n✅ Подготовка завершена!")
    print(f"Скопировано изображений: {total_copied}")
    print(f"Структура создана в: {output_dir}")

    # Выводим статистику
    print_statistics(output_dir, selected_classes)


def print_statistics(output_dir, classes):
    """Выводит статистику по подготовленному датасету"""
    print("\n📊 Статистика датасета:")
    print("=" * 60)

    for split in ['train', 'val', 'test']:
        split_path = os.path.join(output_dir, split)

        if not os.path.exists(split_path):
            print(f"{split}: папка не найдена")
            continue

        total_images = 0
        print(f"\n{split.upper()}:")
        print("-" * 40)

        for class_name in classes:
            class_path = os.path.join(split_path, class_name)
            if os.path.exists(class_path):
                num_images = len([f for f in os.listdir(class_path)
                                  if f.endswith(('.jpg', '.jpeg', '.png'))])
                if num_images > 0:
                    print(f"  {class_name}: {num_images} изображений")
                    total_images += num_images

        print(f"\n  Всего в {split}: {total_images} изображений")

    print("=" * 60)


def prepare_aircraft20_dataset(dataset_path, output_dir='data'):
    """
    Альтернативная функция: создает датасет с 20 классами самолетов,
    которые вы указали в задании
    """
    print("Создание датасета с 20 классами из задания...")

    # 20 классов из вашего задания
    target_classes = [
        'Airbus A320',
        'Su 80',
        'Boeing 737-700',
        'Tupolev Tu-160',
        'Boeing 777-200',
        'MiG-35',
        'Embraer E-Jet E-195',
        'Airbus A330-300',
        'Boeing 787 Dreamliner',
        'MC-21-310',
        'Bombardier CRJ-200',
        'Embraer ERJ-145',
        'Airbus A380 XWB',
        'Sukhoi Superjet 100',
        'ATR 42-200',
        'Bombardier Dash 8 Q300',
        'Yakovlev Yak-42',
        'Tupolev Tu-134',
        'Beechcraft 1900',
        'Airbus A350 XWB'
    ]

    # Убираем дубликаты
    target_classes = list(dict.fromkeys(target_classes))

    print(f"Целевые классы ({len(target_classes)}):")
    for i, cls in enumerate(target_classes):
        print(f"  {i + 1:2d}. {cls}")

    # Так как в FGVC датасете другие названия классов, нужно сопоставить
    # Для начала просто используем классы из датасета, которые наиболее похожи
    print("\nПоиск соответствий в FGVC датасете...")

    # Получим все классы из FGVC
    images_dir = os.path.join(dataset_path, 'data', 'images')
    train_csv = os.path.join(dataset_path, 'train.csv')

    if not os.path.exists(train_csv):
        print("Ошибка: train.csv не найден!")
        return

    df = pd.read_csv(train_csv, header=None)
    fgvc_classes = sorted(df[1].unique())

    print(f"Найдено классов в FGVC: {len(fgvc_classes)}")

    # Ищем совпадения
    matches = {}
    for target in target_classes:
        target_lower = target.lower()
        # Ищем частичные совпадения
        for fgvc_class in fgvc_classes:
            fgvc_lower = fgvc_class.lower()
            if (target_lower in fgvc_lower) or (fgvc_lower in target_lower):
                matches[target] = fgvc_class
                print(f"  ✓ {target} -> {fgvc_class}")
                break

    print(f"\nНайдено совпадений: {len(matches)}/{len(target_classes)}")

    if len(matches) < 5:  # Если слишком мало совпадений
        print("\nСовпадений мало. Используем первые 20 классов из FGVC:")
        selected_fgvc_classes = fgvc_classes[:20]
        for i, cls in enumerate(selected_fgvc_classes):
            print(f"  {i + 1:2d}. {cls}")

        # Создаем mapping
        matches = {f"Class_{i + 1}": cls for i, cls in enumerate(selected_fgvc_classes)}

    # Теперь подготовим датасет с этими классами
    # Используем найденные классы из FGVC
    fgvc_class_list = list(matches.values())
    print(f"\nИспользуем классы из FGVC: {fgvc_class_list}")

    # Подготавливаем датасет только с этими классами
    prepare_fgvc_dataset(dataset_path, output_dir, specific_classes=fgvc_class_list)


def prepare_fgvc_dataset_specific(dataset_path, output_dir='data', specific_classes=None):
    """
    Подготавливает датасет только с указанными классами
    """
    if not specific_classes:
        print("Ошибка: не указаны классы!")
        return

    print(f"Подготовка датасета с классами: {specific_classes}")

    # Пути к файлам
    images_dir = os.path.join(dataset_path, 'data', 'images')
    train_csv = os.path.join(dataset_path, 'train.csv')
    val_csv = os.path.join(dataset_path, 'val.csv')

    if not os.path.exists(images_dir):
        print(f"Ошибка: папка {images_dir} не найдена!")
        return

    # Создаем структуру папок
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    total_copied = 0

    for csv_file, split_name in [(train_csv, 'train'), (val_csv, 'val')]:
        if not os.path.exists(csv_file):
            print(f"Внимание: файл {csv_file} не найден")
            continue

        print(f"\nОбработка {split_name}...")
        df = pd.read_csv(csv_file, header=None)

        for _, row in tqdm(df.iterrows(), total=len(df)):
            if len(row) >= 2:
                image_name = str(row[0]).strip()
                class_name = str(row[1]).strip()

                # Пропускаем если класс не в списке
                if class_name not in specific_classes:
                    continue

                # Создаем папку для класса
                class_output_dir = os.path.join(output_dir, split_name, class_name)
                os.makedirs(class_output_dir, exist_ok=True)

                # Ищем реальный файл
                def find_real_file(base_name):
                    for ext in ['', '.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                        potential_path = os.path.join(images_dir, f"{base_name}{ext}")

                        if os.path.exists(potential_path):
                            if os.path.islink(potential_path):
                                real_path = os.path.realpath(potential_path)
                                if os.path.exists(real_path):
                                    return real_path
                            else:
                                return potential_path

                    # Поиск по части имени
                    for root, dirs, files in os.walk(images_dir):
                        for file in files:
                            if file.startswith(base_name) or base_name in file:
                                full_path = os.path.join(root, file)
                                if os.path.isfile(full_path) and not os.path.islink(full_path):
                                    return full_path

                    return None

                # Ищем реальный файл
                base_name = image_name.split('.')[0] if '.' in image_name else image_name
                src_path = find_real_file(base_name)

                if src_path and os.path.exists(src_path):
                    dst_filename = os.path.basename(src_path)
                    dst_path = os.path.join(class_output_dir, dst_filename)

                    try:
                        shutil.copy2(src_path, dst_path)
                        total_copied += 1
                    except Exception as e:
                        print(f"  Ошибка копирования {src_path}: {e}")
                else:
                    print(f"  ⚠ Файл не найден: {image_name}")

    print(f"\n✅ Подготовка завершена!")
    print(f"Скопировано изображений: {total_copied}")
    print(f"Структура создана в: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Подготовка FGVC Aircraft Dataset')
    parser.add_argument('--dataset', type=str, default='fgvc-aircraft-2013b',
                        help='Путь к датасету FGVC')
    parser.add_argument('--output', type=str, default='data',
                        help='Выходная папка')
    parser.add_argument('--classes', type=int, default=20,
                        help='Количество классов (0 = все)')
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['auto', 'custom20', 'all'],
                        help='Режим подготовки: auto, custom20, all')

    args = parser.parse_args()

    if args.mode == 'custom20':
        # Используем 20 классов из задания
        prepare_aircraft20_dataset(args.dataset, args.output)
    else:
        # Автоматический режим
        prepare_fgvc_dataset(args.dataset, args.output,
                             num_classes=args.classes if args.classes > 0 else None)


if __name__ == "__main__":
    main()