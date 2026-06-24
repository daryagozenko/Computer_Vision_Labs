import cv2
import numpy as np
from typing import Tuple


class ImageSegmentation:
    """Класс для сегментации изображений методом кластеризации"""

    @staticmethod
    def kmeans_segmentation(image: np.ndarray,
                            n_clusters: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Сегментация изображения с помощью K-means кластеризации

        Args:
            image: Входное изображение
            n_clusters: Количество кластеров (от 2 до 20)

        Returns:
            Tuple: (сегментированное изображение, маска кластеров)
        """
        if image is None:
            return None, None

        # Ограничение количества кластеров
        n_clusters = max(2, min(20, n_clusters))

        print(f"Применение K-means сегментации с {n_clusters} кластерами...")

        # Преобразование изображения в 2D массив пикселей
        pixel_data = image.reshape((-1, 3))
        pixel_data = np.float32(pixel_data)

        # Критерии остановки алгоритма
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        # Применение K-means
        _, labels, centers = cv2.kmeans(pixel_data, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Преобразование центров обратно в uint8
        centers = np.uint8(centers)

        # Создание сегментированного изображения
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape(image.shape)

        print(f"Сегментация завершена. Размер изображения: {segmented_image.shape}")
        return segmented_image, labels.reshape(image.shape[:2])

    @staticmethod
    def apply_colormap_to_segmentation(labels: np.ndarray) -> np.ndarray:
        """
        Применение цветовой карты к результатам сегментации
        для лучшей визуализации

        Args:
            labels: Маска кластеров

        Returns:
            Изображение с примененной цветовой картой
        """
        if labels is None:
            return None

        # Нормализация меток для применения цветовой карты
        normalized_labels = cv2.normalize(labels, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Применение цветовой карты
        colored_segmentation = cv2.applyColorMap(normalized_labels, cv2.COLORMAP_JET)

        return colored_segmentation