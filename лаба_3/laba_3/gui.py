import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List

import cv2
from PIL import Image, ImageTk
import numpy as np
from image_processor import ImageProcessor
from edge_detectors import EdgeDetectors
from segmentation import ImageSegmentation
from keypoint_detectors import KeypointDetectors
from license_plate_detector import LicensePlateDetector


class ImageEditorGUI:
    """Графический интерфейс для редактора изображений"""

    def __init__(self, root):
        self.root = root
        self.root.title("Редактор изображений - Анализ и обработка")
        self.root.geometry("1200x800")

        # Инициализация процессора изображений
        self.processor = ImageProcessor()

        # Переменные интерфейса
        self.n_clusters = tk.IntVar(value=5)
        self.canny_low_threshold = tk.IntVar(value=50)
        self.canny_high_threshold = tk.IntVar(value=150)
        self.roberts_threshold = tk.DoubleVar(value=30.0)

        self.setup_ui()

    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        # Основной фрейм
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Панель управления слева
        control_frame = ttk.LabelFrame(main_frame, text="Управление", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Кнопка загрузки изображения
        ttk.Button(control_frame, text="Загрузить изображение",
                   command=self.load_image).pack(fill=tk.X, pady=5)

        # Кнопка сброса настроек
        ttk.Button(control_frame, text="Сбросить настройки",
                   command=self.reset_settings).pack(fill=tk.X, pady=5)

        # Разделитель
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Детекторы границ
        edge_frame = ttk.LabelFrame(control_frame, text="Детекторы границ", padding=5)
        edge_frame.pack(fill=tk.X, pady=5)

        ttk.Button(edge_frame, text="Детектор Кэнни",
                   command=self.apply_canny).pack(fill=tk.X, pady=2)

        # Настройки Кэнни - поля ввода вместо ползунков
        canny_settings = ttk.Frame(edge_frame)
        canny_settings.pack(fill=tk.X, pady=2)

        ttk.Label(canny_settings, text="Нижний порог:").grid(row=0, column=0, sticky=tk.W)
        self.canny_low_entry = ttk.Entry(canny_settings, width=8, textvariable=self.canny_low_threshold)
        self.canny_low_entry.grid(row=0, column=1, padx=5)

        ttk.Label(canny_settings, text="Верхний порог:").grid(row=1, column=0, sticky=tk.W)
        self.canny_high_entry = ttk.Entry(canny_settings, width=8, textvariable=self.canny_high_threshold)
        self.canny_high_entry.grid(row=1, column=1, padx=5)

        # Валидация для полей ввода
        self.canny_low_entry.bind('<FocusOut>', self.validate_canny_thresholds)
        self.canny_high_entry.bind('<FocusOut>', self.validate_canny_thresholds)

        ttk.Button(edge_frame, text="Оператор Робертса",
                   command=self.apply_roberts).pack(fill=tk.X, pady=2)

        # Настройки Робертса
        roberts_settings = ttk.Frame(edge_frame)
        roberts_settings.pack(fill=tk.X, pady=2)
        ttk.Label(roberts_settings, text="Порог:").pack(side=tk.LEFT)
        self.roberts_entry = ttk.Entry(roberts_settings, width=8, textvariable=self.roberts_threshold)
        self.roberts_entry.pack(side=tk.LEFT, padx=5)

        # Сегментация
        seg_frame = ttk.LabelFrame(control_frame, text="Сегментация", padding=5)
        seg_frame.pack(fill=tk.X, pady=5)

        ttk.Label(seg_frame, text="Количество кластеров:").pack(anchor=tk.W)

        # Поле ввода для кластеров вместо ползунка
        cluster_frame = ttk.Frame(seg_frame)
        cluster_frame.pack(fill=tk.X, pady=2)
        self.cluster_entry = ttk.Entry(cluster_frame, width=8, textvariable=self.n_clusters)
        self.cluster_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(cluster_frame, text="Применить",
                   command=self.apply_segmentation).pack(side=tk.LEFT, padx=5)

        # Валидация для кластеров
        self.cluster_entry.bind('<FocusOut>', self.validate_clusters)

        # Ключевые точки
        kp_frame = ttk.LabelFrame(control_frame, text="Ключевые точки", padding=5)
        kp_frame.pack(fill=tk.X, pady=5)

        ttk.Button(kp_frame, text="SIFT",
                   command=lambda: self.apply_keypoint_detection('sift')).pack(fill=tk.X, pady=2)
        ttk.Button(kp_frame, text="FAST",
                   command=lambda: self.apply_keypoint_detection('fast')).pack(fill=tk.X, pady=2)

        # Детектор номеров
        plate_frame = ttk.LabelFrame(control_frame, text="Детектор номеров", padding=5)
        plate_frame.pack(fill=tk.X, pady=5)

        ttk.Button(plate_frame, text="Обнаружить номер",
                   command=self.detect_license_plate).pack(fill=tk.X, pady=2)

        # Область отображения изображений
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Исходное изображение
        self.original_label = ttk.Label(display_frame, text="Исходное изображение")
        self.original_label.pack(pady=5)
        self.original_canvas = tk.Canvas(display_frame, width=400, height=300, bg='white')
        self.original_canvas.pack(pady=5, fill=tk.BOTH, expand=True)

        # Обработанное изображение
        self.processed_label = ttk.Label(display_frame, text="Обработанное изображение")
        self.processed_label.pack(pady=5)
        self.processed_canvas = tk.Canvas(display_frame, width=400, height=300, bg='white')
        self.processed_canvas.pack(pady=5, fill=tk.BOTH, expand=True)

    def validate_canny_thresholds(self, event=None):
        """Валидация порогов Кэнни"""
        try:
            low = int(self.canny_low_threshold.get())
            high = int(self.canny_high_threshold.get())

            # Проверка диапазонов
            if low < 0:
                low = 0
            if high > 255:
                high = 255
            if low >= high:
                low = high - 1 if high > 0 else 0

            self.canny_low_threshold.set(low)
            self.canny_high_threshold.set(high)

        except ValueError:
            # Если введено не число, сбрасываем к значениям по умолчанию
            self.canny_low_threshold.set(50)
            self.canny_high_threshold.set(150)
            messagebox.showerror("Ошибка", "Пороги должны быть целыми числами")

    def validate_clusters(self, event=None):
        """Валидация количества кластеров"""
        try:
            clusters = int(self.n_clusters.get())

            # Проверка диапазона
            if clusters < 2:
                clusters = 2
            elif clusters > 20:
                clusters = 20

            self.n_clusters.set(clusters)

        except ValueError:
            # Если введено не число, сбрасываем к значению по умолчанию
            self.n_clusters.set(5)
            messagebox.showerror("Ошибка", "Количество кластеров должно быть целым числом")

    def reset_settings(self):
        """Сброс всех настроек к значениям по умолчанию и изображения к оригиналу"""
        # Сброс настроек
        self.canny_low_threshold.set(50)
        self.canny_high_threshold.set(150)
        self.roberts_threshold.set(30.0)
        self.n_clusters.set(5)

        # Сброс изображения к оригиналу
        self.processor.reset_processed_image()
        self.display_images()

    def load_image(self):
        """Загрузка изображения"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )

        if file_path:
            if self.processor.load_image(file_path):
                self.display_images()
            else:
                messagebox.showerror("Ошибка", "Не удалось загрузить изображение!")

    def reset_image(self):
        """Сброс изображения к оригиналу"""
        self.processor.reset_processed_image()
        self.display_images()

    def display_images(self):
        """Отображение изображений на канвасах"""
        # Оригинальное изображение
        original = self.processor.get_original_image()
        if original is not None:
            original_display = self.prepare_image_for_display(original)
            self.display_image_on_canvas(original_display, self.original_canvas)

        # Обработанное изображение
        processed = self.processor.get_processed_image()
        if processed is not None:
            processed_display = self.prepare_image_for_display(processed)
            self.display_image_on_canvas(processed_display, self.processed_canvas)

    def prepare_image_for_display(self, image: np.ndarray, max_size: tuple = (400, 300)) -> ImageTk.PhotoImage:
        """Подготовка изображения для отображения в Tkinter"""
        # Конвертация BGR в RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Изменение размера если нужно
        h, w = image_rgb.shape[:2]
        if h > max_size[1] or w > max_size[0]:
            scale = min(max_size[0] / w, max_size[1] / h)
            new_w, new_h = int(w * scale), int(h * scale)
            image_resized = cv2.resize(image_rgb, (new_w, new_h))
        else:
            image_resized = image_rgb

        # Конвертация в формат для Tkinter
        pil_image = Image.fromarray(image_resized)
        return ImageTk.PhotoImage(pil_image)

    def display_image_on_canvas(self, photo_image: ImageTk.PhotoImage, canvas: tk.Canvas):
        """Отображение изображения на канвасе"""
        canvas.delete("all")
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:  # Если канвас уже отрисован
            x = canvas_width // 2
            y = canvas_height // 2
            canvas.create_image(x, y, image=photo_image)
            canvas.image = photo_image  # Сохраняем ссылку

    def apply_canny(self):
        """Применение детектора Кэнни"""
        if self.processor.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение!")
            return

        # Валидация порогов перед применением
        self.validate_canny_thresholds()

        edges = EdgeDetectors.canny_edge_detector(
            self.processor.original_image,
            self.canny_low_threshold.get(),
            self.canny_high_threshold.get()
        )

        self.processor.processed_image = edges
        self.display_images()

    def apply_roberts(self):
        """Применение оператора Робертса"""
        if self.processor.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение!")
            return

        edges = EdgeDetectors.roberts_edge_detector(
            self.processor.original_image,
            self.roberts_threshold.get()
        )

        self.processor.processed_image = edges
        self.display_images()

    def apply_segmentation(self):
        """Применение сегментации"""
        if self.processor.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение!")
            return

        # Валидация кластеров перед применением
        self.validate_clusters()

        print(f"Запуск сегментации с {self.n_clusters.get()} кластерами...")

        try:
            segmented_image, labels = ImageSegmentation.kmeans_segmentation(
                self.processor.original_image,
                self.n_clusters.get()
            )

            if segmented_image is not None:
                self.processor.processed_image = segmented_image
                self.display_images()
                messagebox.showinfo("Сегментация",
                                    f"Сегментация на {self.n_clusters.get()} кластеров выполнена успешно!")
            else:
                messagebox.showerror("Ошибка", "Не удалось выполнить сегментацию")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при сегментации: {str(e)}")
            print(f"Ошибка сегментации: {e}")

    def apply_keypoint_detection(self, method: str):
        """Применение детектора ключевых точек"""
        if self.processor.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение!")
            return

        if method == 'sift':
            result_image, keypoints = KeypointDetectors.sift_keypoints(
                self.processor.original_image
            )
        elif method == 'surf':
            result_image, keypoints = KeypointDetectors.surf_keypoints(
                self.processor.original_image
            )
        elif method == 'fast':
            result_image, keypoints = KeypointDetectors.fast_keypoints(
                self.processor.original_image
            )
        else:
            return

        self.processor.processed_image = result_image
        self.display_images()

        messagebox.showinfo("Ключевые точки",
                            f"Обнаружено {len(keypoints)} ключевых точек методом {method.upper()}")

    def detect_license_plate(self):
        """Обнаружение автомобильного номера с показом всех найденных номеров"""
        if self.processor.original_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение!")
            return

        # Используем улучшенный детектор, который возвращает все номера
        image_with_boxes, plates = LicensePlateDetector.detect_license_plate(
            self.processor.original_image
        )

        if plates:
            self.processor.processed_image = image_with_boxes
            self.display_images()

            # Показываем все найденные номера
            self.show_all_license_plates(plates)

        else:
            messagebox.showinfo("Результат", "Автомобильные номера не обнаружены")
            self.processor.processed_image = self.processor.original_image.copy()
            self.display_images()

    def show_all_license_plates(self, plates: List[np.ndarray]):
        """Показать все найденные номера в отдельном окне"""
        plates_window = tk.Toplevel(self.root)
        plates_window.title(f"Обнаруженные номера ({len(plates)} шт.)")
        plates_window.geometry("500x400")

        # Заголовок
        title_label = ttk.Label(plates_window,
                                text=f"Найдено номеров: {len(plates)}",
                                font=('Arial', 12, 'bold'))
        title_label.pack(pady=10)

        # Фрейм для прокрутки
        container = ttk.Frame(plates_window)
        container.pack(fill=tk.BOTH, expand=True, padx=10)

        # Холст с прокруткой
        canvas = tk.Canvas(container)
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Отображаем каждый номер
        for i, plate in enumerate(plates):
            plate_frame = ttk.LabelFrame(scrollable_frame, text=f"Номер {i + 1}", padding=10)
            plate_frame.pack(fill=tk.X, pady=5, padx=5)

            # Подготавливаем изображение для отображения (ограничиваем размер)
            plate_display = self.prepare_image_for_display(plate, (300, 100))

            # Создаем метку с изображением
            plate_label = ttk.Label(plate_frame, image=plate_display)
            plate_label.image = plate_display
            plate_label.pack(pady=5)

            # Информация о размере
            info_text = f"Размер: {plate.shape[1]}×{plate.shape[0]} пикселей"
            info_label = ttk.Label(plate_frame, text=info_text)
            info_label.pack()

        # Упаковываем canvas и scrollbar
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Кнопка закрытия
        ttk.Button(plates_window, text="Закрыть",
                   command=plates_window.destroy).pack(pady=10)

        # Центрируем окно
        plates_window.transient(self.root)
        plates_window.grab_set()
        plates_window.focus_force()

    def show_license_plate(self, license_plate: np.ndarray):
        """Показать обрезанный номер в отдельном окне"""
        plate_window = tk.Toplevel(self.root)
        plate_window.title("Обнаруженный номер")
        plate_window.geometry("300x200")

        plate_image = self.prepare_image_for_display(license_plate, (250, 150))
        plate_label = ttk.Label(plate_window, image=plate_image)
        plate_label.image = plate_image
        plate_label.pack(pady=10)

        ttk.Button(plate_window, text="Закрыть",
                   command=plate_window.destroy).pack(pady=5)