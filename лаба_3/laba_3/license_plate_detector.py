
import cv2
import numpy as np
from typing import Tuple, List


class LicensePlateDetector:
    """Класс для обнаружения автомобильных номеров"""

    @staticmethod
    def detect_license_plate(image: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Универсальный метод - пробуем разные подходы
        """
        if image is None:
            return None, []

        # Пробуем метод 1 (простой)
        result1, plates1 = LicensePlateDetector._method_simple(image)
        if plates1:
            return result1, plates1

        # Пробуем метод 2 (морфологический)
        # result2, plates2 = LicensePlateDetector._method_morphological(image)
        # if plates2:
        #     return result2, plates2

        return image, []

    @staticmethod
    def _method_simple(image: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Простой метод через Canny и контуры"""
        result_image = image.copy()
        plates = []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        plate_count = 0
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h

                if 2.0 < aspect_ratio < 5.0:
                    plate_count += 1
                    cv2.drawContours(result_image, [approx], -1, (0, 255, 0), 3)
                    cv2.putText(result_image, f"№{plate_count}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Обрезка с отступами
                    padding = 5
                    x_pad = max(0, x - padding)
                    y_pad = max(0, y - padding)
                    w_pad = min(image.shape[1] - x_pad, w + 2 * padding)
                    h_pad = min(image.shape[0] - y_pad, h + 2 * padding)

                    plate = image[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
                    plates.append(plate)

        return result_image, plates