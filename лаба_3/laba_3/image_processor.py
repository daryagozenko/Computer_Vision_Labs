import numpy as np
from typing import Optional
import os
from PIL import Image, ImageFile
import cv2

# Разрешаем загрузку поврежденных изображений
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageProcessor:
    """Базовый класс для обработки изображений"""

    def __init__(self):
        self.original_image: Optional[np.ndarray] = None
        self.processed_image: Optional[np.ndarray] = None

    def load_image(self, image_path: str) -> bool:
        """Загрузка изображения из файла через PIL"""
        try:
            # Нормализация пути
            image_path = os.path.normpath(image_path)

            # Проверка существования файла
            if not os.path.exists(image_path):
                print(f"Файл не существует: {image_path}")
                return False

            # Проверка размера файла
            file_size = os.path.getsize(image_path)
            if file_size == 0:
                print("Файл пустой")
                return False

            print(f"Загрузка изображения: {image_path}")
            print(f"Размер файла: {file_size} байт")

            # Загрузка через PIL
            success = self._load_with_pil(image_path)

            if success:
                print(f"Изображение успешно загружено через PIL")
                print(f"Размер: {self.original_image.shape}")
                print(f"Тип данных: {self.original_image.dtype}")
                return True
            else:
                print("Не удалось загрузить изображение через PIL")
                return False

        except Exception as e:
            print(f"Критическая ошибка загрузки: {e}")
            return False

    def _load_with_pil(self, image_path: str) -> bool:
        """Загрузка изображения через PIL с конвертацией в формат OpenCV"""
        try:
            # Открываем изображение через PIL
            with Image.open(image_path) as pil_image:
                print(f"PIL обнаружил формат: {pil_image.format}")
                print(f"Исходный режим: {pil_image.mode}")
                print(f"Исходный размер: {pil_image.size}")

                # Конвертируем в RGB если нужно (убираем альфа-канал)
                if pil_image.mode in ('RGBA', 'LA', 'P'):
                    # Для изображений с прозрачностью
                    if pil_image.mode == 'P':
                        pil_image = pil_image.convert('RGBA')

                    # Создаем белый фон и накладываем изображение
                    background = Image.new('RGB', pil_image.size, (255, 255, 255))
                    if pil_image.mode == 'RGBA':
                        background.paste(pil_image, mask=pil_image.split()[-1])
                    else:
                        background.paste(pil_image)
                    pil_image = background
                elif pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')

                print(f"Конвертированный режим: {pil_image.mode}")

                # Конвертируем PIL Image в numpy array
                numpy_image = np.array(pil_image)
                print(f"Numpy array форма: {numpy_image.shape}")

                # Конвертируем RGB в BGR для OpenCV
                if len(numpy_image.shape) == 3 and numpy_image.shape[2] == 3:
                    numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
                    print("Конвертирован RGB → BGR")

                self.original_image = numpy_image
                self.processed_image = numpy_image.copy()
                return True

        except Exception as e:
            print(f"Ошибка загрузки через PIL: {e}")
            return False

    def get_original_image(self) -> Optional[np.ndarray]:
        """Получить оригинальное изображение"""
        return self.original_image

    def get_processed_image(self) -> Optional[np.ndarray]:
        """Получить обработанное изображение"""
        return self.processed_image

    def reset_processed_image(self):
        """Сбросить обработанное изображение к оригинальному"""
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            print("Изображение сброшено к оригиналу")