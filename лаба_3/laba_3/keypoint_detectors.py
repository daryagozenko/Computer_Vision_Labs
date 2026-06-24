import cv2
import numpy as np
from typing import List, Tuple


class KeypointDetectors:
    """Класс для обнаружения ключевых точек различными методами"""

    @staticmethod
    def sift_keypoints(image: np.ndarray,
                       max_keypoints: int = 100) -> Tuple[np.ndarray, List]:
        """
        Обнаружение ключевых точек с помощью SIFT

        Args:
            image: Входное изображение
            max_keypoints: Максимальное количество ключевых точек

        Returns:
            Tuple: (изображение с точками, список ключевых точек)
        """
        if image is None:
            return None, []

        # Конвертация в оттенки серого
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Инициализация SIFT детектора
        sift = cv2.SIFT_create(nfeatures=max_keypoints)

        # Обнаружение ключевых точек и дескрипторов
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        # Рисование ключевых точек на изображении
        result_image = cv2.drawKeypoints(
            image, keypoints, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        return result_image, keypoints



    @staticmethod
    def fast_keypoints(image: np.ndarray,
                       threshold: int = 50,
                       nonmax_suppression: bool = True) -> Tuple[np.ndarray, List]:
        """
        Обнаружение ключевых точек с помощью FAST

        Args:
            image: Входное изображение
            threshold: Порог для детектирования
            nonmax_suppression: Подавление немаксимумов

        Returns:
            Tuple: (изображение с точками, список ключевых точек)
        """
        if image is None:
            return None, []

        # Конвертация в оттенки серого
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Инициализация FAST детектора
        fast = cv2.FastFeatureDetector_create(threshold)
        fast.setNonmaxSuppression(nonmax_suppression)

        # Обнаружение ключевых точек
        keypoints = fast.detect(gray, None)

        # Рисование ключевых точек
        result_image = cv2.drawKeypoints(
            image, keypoints, None, color=(255, 0, 0)
        )

        return result_image, keypoints