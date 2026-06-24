# check_images.py
import os
from PIL import Image
import shutil


def check_and_fix_images(data_dir='data'):
    """Проверяет и исправляет проблемные изображения"""

    print("🔍 Проверка изображений...")

    problematic_files = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                file_path = os.path.join(root, file)

                try:
                    # Пробуем открыть изображение
                    with Image.open(file_path) as img:
                        img.verify()  # Проверяем целостность

                    # Проверяем размер
                    with Image.open(file_path) as img:
                        if img.size[0] == 0 or img.size[1] == 0:
                            problematic_files.append((file_path, "Нулевой размер"))

                except Exception as e:
                    problematic_files.append((file_path, str(e)))

    print(f"\nНайдено проблемных файлов: {len(problematic_files)}")

    if problematic_files:
        print("\nСписок проблемных файлов:")
        for file_path, error in problematic_files[:10]:  # показываем первые 10
            print(f"  ❌ {file_path}")
            print(f"     Ошибка: {error}")

            # Предлагаем удалить или переименовать
            print("     Действия:")
            print("     1. Удалить файл")
            print("     2. Переименовать (убрать .jpg)")
            print("     3. Пропустить")

            choice = input("     Выберите действие (1-3): ").strip()

            if choice == '1':
                os.remove(file_path)
                print(f"     ✅ Файл удален: {file_path}")
            elif choice == '2':
                # Убираем .jpg если файл не jpg
                new_path = file_path.rsplit('.', 1)[0] + '_fixed'
                shutil.move(file_path, new_path)
                print(f"     ✅ Файл переименован: {new_path}")

    return len(problematic_files)


def find_files_without_extension(data_dir='data'):
    """Находит файлы без расширения или с неправильным расширением"""

    print("\n🔍 Поиск файлов без расширения .jpg...")

    suspicious_files = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)

            # Ищем файлы без .jpg/.jpeg/.png
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Проверяем, может это изображение без расширения?
                try:
                    with Image.open(file_path) as img:
                        # Если открылось - это изображение без правильного расширения
                        suspicious_files.append(file_path)
                except:
                    pass  # Не изображение

    if suspicious_files:
        print(f"\nНайдено файлов без правильного расширения: {len(suspicious_files)}")
        for file_path in suspicious_files[:10]:
            print(f"  ⚠ {file_path}")

            # Пробуем определить тип
            try:
                with Image.open(file_path) as img:
                    print(f"     Формат: {img.format}, Размер: {img.size}")

                    # Предлагаем добавить расширение
                    new_path = file_path + '.jpg'
                    os.rename(file_path, new_path)
                    print(f"     ✅ Переименован в: {new_path}")
            except:
                print(f"     Не удалось определить тип")

    return len(suspicious_files)


if __name__ == "__main__":
    print("=" * 60)
    print("ПРОВЕРКА И ИСПРАВЛЕНИЕ ИЗОБРАЖЕНИЙ")
    print("=" * 60)

    # Проверяем проблемные изображения
    problem_count = check_and_fix_images()

    # Ищем файлы без расширения
    no_ext_count = find_files_without_extension()

    print("\n" + "=" * 60)
    if problem_count == 0 and no_ext_count == 0:
        print("✅ Все изображения в порядке!")
    else:
        print(f"⚠ Найдено проблем: {problem_count + no_ext_count}")
        print("Рекомендуется запустить обучение после исправлений")