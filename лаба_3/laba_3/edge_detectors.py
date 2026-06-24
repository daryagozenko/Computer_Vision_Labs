import cv2
import numpy as np


class EdgeDetectors:
    """Класс для детектирования границ различными методами"""

    @staticmethod
    def canny_edge_detector(image: np.ndarray,
                            low_threshold: int = 50,
                            high_threshold: int = 150) -> np.ndarray:
        """
        Детектор границ Кэнни

        Args:
            image: Входное изображение
            low_threshold: Нижний порог
            high_threshold: Верхний порог

        Returns:
            Изображение с обнаруженными границами
        """
        if image is None:
            return None

        # Конвертация в оттенки серого если нужно
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Применение детектора Кэнни
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        return edges

    @staticmethod
    def roberts_edge_detector(image: np.ndarray,
                              threshold: float = 30.0) -> np.ndarray:
        """
        Перекрестный оператор Робертса

        Args:
            image: Входное изображение
            threshold: Порог для бинаризации

        Returns:
            Изображение с обнаруженными границами
        """
        if image is None:
            return None

        # Конвертация в оттенки серого если нужно
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Оператор Робертса
        kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

        # Применение свертки
        gx = cv2.filter2D(gray.astype(np.float32), -1, kernel_x)
        gy = cv2.filter2D(gray.astype(np.float32), -1, kernel_y)

        # Вычисление силы градиента
        gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)

        # Применение порога
        edges = (gradient_magnitude > threshold).astype(np.uint8) * 255

        return edges

    @staticmethod
    def sobel_edge_detector(image: np.ndarray,
                            threshold: float = 50.0) -> np.ndarray:
        """
        Оператор Собеля (дополнительная функция)

        Args:
            image: Входное изображение
            threshold: Порог для бинаризации

        Returns:
            Изображение с обнаруженными границами
        """
        if image is None:
            return None

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Оператор Собеля
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Вычисление силы градиента
        gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # Нормализация и порог
        edges = (gradient_magnitude > threshold).astype(np.uint8) * 255

        return edges