from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
from ImageProcessor import ImageProcessor


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Обработчик изображений")
        self.root.geometry("1400x900")

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.processor = ImageProcessor()
        self.create_widgets()
        self.current_histogram_channel = 'all'
        self.last_update_time = 0
        self.update_delay = 100
        self.update_pending = False

    def on_closing(self):
        self.processor.close_images()
        self.root.destroy()

    def on_slider_change(self, event=None):
        """Ограничение частоты обновлений для плавности"""
        current_time = time.time() * 1000
        if current_time - self.last_update_time > self.update_delay:
            self.last_update_time = current_time
            self.update_image()

    def toggle_grayscale(self):
        """Переключение между grayscale и RGB режимами"""
        if self.processor.is_grayscale():
            self.processor.convert_to_rgb()
            self.grayscale_button.config(text="В градации серого")
            self.status_var.set("Изображение преобразовано в RGB")
            # Восстанавливаем значения насыщенности
            self.saturation_var.set(int(self.processor.current_saturation * 50))
            self.saturation_label.config(text=str(int(self.processor.current_saturation * 50)))
        else:
            self.processor.convert_grayscale()
            self.grayscale_button.config(text="В RGB")
            self.status_var.set("Изображение преобразовано в градации серого")
            # Сбрасываем насыщенность для grayscale
            self.saturation_var.set(0)
            self.saturation_label.config(text="0")

        self.display_images()
        self.update_histogram()
        self.adjust_image()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        left_panel = ttk.LabelFrame(main_frame, text="Управление", padding="10")
        left_panel.grid(row=0, column=0, rowspan=2, sticky=(tk.N, tk.S), padx=(0, 10))

        # Кнопки управления
        ttk.Button(left_panel, text="Загрузить изображение", command=self.open_image).grid(row=0, column=0, pady=5,
                                                                                           sticky=(tk.W, tk.E))
        ttk.Button(left_panel, text="Показать информацию", command=self.show_info).grid(row=1, column=0, pady=5,
                                                                                        sticky=(tk.W, tk.E))
        self.grayscale_button = ttk.Button(left_panel, text="В градации серого", command=self.toggle_grayscale)
        self.grayscale_button.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E))
        ttk.Button(left_panel, text="Повернуть на 90°", command=self.rotate_90).grid(row=3, column=0, pady=5,
                                                                                     sticky=(tk.W, tk.E))
        ttk.Button(left_panel, text="Сбросить настройки", command=self.reset_settings).grid(row=4, column=0, pady=5,
                                                                                            sticky=(tk.W, tk.E))
        ttk.Button(left_panel, text="Сохранить изображение", command=self.save_image).grid(row=5, column=0, pady=5,
                                                                                           sticky=(tk.W, tk.E))

        # Коррекция для черно-белых изображений
        ttk.Label(left_panel, text="Линейная коррекция (a):").grid(row=6, column=0, pady=(15, 0), sticky=tk.W)
        self.linear_a_var = tk.DoubleVar(value=1.0)
        linear_a_scale = ttk.Scale(left_panel, from_=0.1, to=3.0, variable=self.linear_a_var,
                                   command=self.apply_linear_correction, orient=tk.HORIZONTAL)
        linear_a_scale.grid(row=7, column=0, pady=5, sticky=(tk.W, tk.E))

        ttk.Label(left_panel, text="Гамма-коррекция:").grid(row=8, column=0, pady=(15, 0), sticky=tk.W)
        self.gamma_var = tk.DoubleVar(value=1.0)
        gamma_scale = ttk.Scale(left_panel, from_=0.1, to=3.0, variable=self.gamma_var,
                                command=self.apply_gamma_correction, orient=tk.HORIZONTAL)
        gamma_scale.grid(row=9, column=0, pady=5, sticky=(tk.W, tk.E))

        # Основные слайдеры
        ttk.Label(left_panel, text="Яркость:").grid(row=10, column=0, pady=(15, 0), sticky=tk.W)
        self.brightness_var = tk.IntVar(value=50)
        brightness_frame = ttk.Frame(left_panel)
        brightness_frame.grid(row=11, column=0, pady=5, sticky=(tk.W, tk.E))
        brightness_scale = ttk.Scale(brightness_frame, from_=0, to=100, variable=self.brightness_var,
                                     command=self.on_scale_update, orient=tk.HORIZONTAL)
        brightness_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.brightness_label = ttk.Label(brightness_frame, text="50", width=4)
        self.brightness_label.pack(side=tk.RIGHT, padx=(5, 0))

        ttk.Label(left_panel, text="Контраст:").grid(row=12, column=0, pady=(15, 0), sticky=tk.W)
        self.contrast_var = tk.IntVar(value=50)
        contrast_frame = ttk.Frame(left_panel)
        contrast_frame.grid(row=13, column=0, pady=5, sticky=(tk.W, tk.E))
        contrast_scale = ttk.Scale(contrast_frame, from_=0, to=100, variable=self.contrast_var,
                                   command=self.on_scale_update, orient=tk.HORIZONTAL)
        contrast_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.contrast_label = ttk.Label(contrast_frame, text="50", width=4)
        self.contrast_label.pack(side=tk.RIGHT, padx=(5, 0))

        ttk.Label(left_panel, text="Насыщенность:").grid(row=14, column=0, pady=(15, 0), sticky=tk.W)
        self.saturation_var = tk.IntVar(value=50)
        saturation_frame = ttk.Frame(left_panel)
        saturation_frame.grid(row=15, column=0, pady=5, sticky=(tk.W, tk.E))
        saturation_scale = ttk.Scale(saturation_frame, from_=0, to=100, variable=self.saturation_var,
                                     command=self.on_scale_update, orient=tk.HORIZONTAL)
        saturation_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.saturation_label = ttk.Label(saturation_frame, text="50", width=4)
        self.saturation_label.pack(side=tk.RIGHT, padx=(5, 0))

        # Выбор канала для гистограммы
        ttk.Label(left_panel, text="Гистограмма канал:").grid(row=16, column=0, pady=(15, 0), sticky=tk.W)
        self.hist_channel = tk.StringVar(value="all")
        channel_combo = ttk.Combobox(left_panel, textvariable=self.hist_channel,
                                     values=["all", "r", "g", "b"], state="readonly")
        channel_combo.grid(row=17, column=0, pady=5, sticky=(tk.W, tk.E))
        channel_combo.bind('<<ComboboxSelected>>', self.update_histogram)

        ttk.Button(left_panel, text="Статистика", command=self.show_stats).grid(row=18, column=0, pady=5,
                                                                                sticky=(tk.W, tk.E))

        # Правая панель с двумя изображениями
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.N, tk.E, tk.W, tk.S))
        right_panel.columnconfigure(0, weight=1)
        right_panel.columnconfigure(1, weight=1)
        right_panel.rowconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)

        # Оригинальное изображение
        orig_frame = ttk.LabelFrame(right_panel, text="Оригинал", padding="5")
        orig_frame.grid(row=0, column=0, sticky=(tk.N, tk.E, tk.W, tk.S), padx=(0, 5), pady=(0, 10))
        orig_frame.columnconfigure(0, weight=1)
        orig_frame.rowconfigure(0, weight=1)

        self.orig_canvas = tk.Canvas(orig_frame, bg='white')
        self.orig_canvas.grid(row=0, column=0, sticky=(tk.N, tk.E, tk.W, tk.S))
        self.orig_canvas.bind('<Configure>', lambda e: self.display_original_image())

        # Обработанное изображение
        proc_frame = ttk.LabelFrame(right_panel, text="Обработанное", padding="5")
        proc_frame.grid(row=0, column=1, sticky=(tk.N, tk.E, tk.W, tk.S), padx=(5, 0), pady=(0, 10))
        proc_frame.columnconfigure(0, weight=1)
        proc_frame.rowconfigure(0, weight=1)

        self.proc_canvas = tk.Canvas(proc_frame, bg='white')
        self.proc_canvas.grid(row=0, column=0, sticky=(tk.N, tk.E, tk.W, tk.S))
        self.proc_canvas.bind('<Configure>', lambda e: self.display_processed_image())

        # Гистограмма
        self.hist_frame = ttk.LabelFrame(right_panel, text="Гистограмма", padding="5")
        self.hist_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.N, tk.E, tk.W, tk.S))
        self.hist_frame.columnconfigure(0, weight=1)
        self.hist_frame.rowconfigure(0, weight=1)

        self.hist_canvas_frame = ttk.Frame(self.hist_frame)
        self.hist_canvas_frame.grid(row=0, column=0, sticky=(tk.N, tk.E, tk.W, tk.S))
        self.hist_canvas_frame.columnconfigure(0, weight=1)
        self.hist_canvas_frame.rowconfigure(0, weight=1)

        # Статус бар
        self.status_var = tk.StringVar(value="Готов к работе")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E))

    def display_original_image(self):
        """Отображение оригинального изображения"""
        if self.processor.original_image:
            try:
                canvas_width = self.orig_canvas.winfo_width()
                canvas_height = self.orig_canvas.winfo_height()

                if canvas_width <= 1 or canvas_height <= 1:
                    return

                img = self.processor.original_image.copy()
                img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)

                self.orig_tk_image = ImageTk.PhotoImage(img)
                self.orig_canvas.delete("all")
                self.orig_canvas.create_image(
                    canvas_width // 2,
                    canvas_height // 2,
                    anchor=tk.CENTER,
                    image=self.orig_tk_image
                )
                img.close()
            except Exception as e:
                print(f"Ошибка отображения оригинала: {e}")

    def display_processed_image(self):
        """Отображение обработанного изображения"""
        if self.processor.processed_image:
            try:
                canvas_width = self.proc_canvas.winfo_width()
                canvas_height = self.proc_canvas.winfo_height()

                if canvas_width <= 1 or canvas_height <= 1:
                    return

                img = self.processor.processed_image.copy()
                img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)

                self.proc_tk_image = ImageTk.PhotoImage(img)
                self.proc_canvas.delete("all")
                self.proc_canvas.create_image(
                    canvas_width // 2,
                    canvas_height // 2,
                    anchor=tk.CENTER,
                    image=self.proc_tk_image
                )
                img.close()
            except Exception as e:
                print(f"Ошибка отображения обработанного: {e}")

    def display_images(self):
        """Отображение обоих изображений"""
        self.display_original_image()
        self.display_processed_image()

    def on_scale_update(self, event):
        """Обработчик изменения слайдеров"""
        self.brightness_label.config(text=f"{self.brightness_var.get()}")
        self.contrast_label.config(text=f"{self.contrast_var.get()}")
        self.saturation_label.config(text=f"{self.saturation_var.get()}")

        current_time = time.time() * 1000
        if current_time - self.last_update_time > self.update_delay:
            self.last_update_time = current_time
            self.adjust_image()
        else:
            if not self.update_pending:
                self.update_pending = True
                self.root.after(self.update_delay, self.delayed_adjust)

    def delayed_adjust(self):
        self.update_pending = False
        self.adjust_image()

    def show_stats(self):
        stats = self.processor.get_image_stats()
        messagebox.showinfo("Статистика изображения", stats)

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            if self.processor.load_image(file_path):
                self.display_images()
                self.update_histogram()
                self.status_var.set(f"Загружено: {os.path.basename(file_path)}")

                # Получаем текущие значения из процессора
                current_values = self.processor.get_current_values()
                self.brightness_var.set(current_values['brightness'])
                self.contrast_var.set(current_values['contrast'])
                self.saturation_var.set(current_values['saturation'])

                # Обновляем метки
                self.brightness_label.config(text=str(current_values['brightness']))
                self.contrast_label.config(text=str(current_values['contrast']))
                self.saturation_label.config(text=str(current_values['saturation']))

                # Обновляем текст кнопки grayscale
                if self.processor.is_grayscale():
                    self.grayscale_button.config(text="В RGB")
                else:
                    self.grayscale_button.config(text="В градации серого")

    def show_info(self):
        info = self.processor.get_image_info()
        if info:
            info_text = "Информация об изображении:\n\n"
            for key, value in info.items():
                if key == 'EXIF данные':
                    info_text += f"{key}:\n"
                    for exif_key, exif_value in value.items():
                        info_text += f"  {exif_key}: {exif_value}\n"
                else:
                    info_text += f"{key}: {value}\n"

            info_window = tk.Toplevel(self.root)
            info_window.title("Информация об изображении")
            info_window.geometry("500x400")

            text_widget = tk.Text(info_window, wrap=tk.WORD)
            text_widget.insert(tk.END, info_text)
            text_widget.config(state=tk.DISABLED)
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def adjust_image(self, event=None):
        if self.processor.original_image:
            try:
                # Используем отдельные методы для каждого параметра
                self.processor.adjust_brightness(self.brightness_var.get())
                self.processor.adjust_contrast(self.contrast_var.get())
                self.processor.adjust_saturation(self.saturation_var.get())

                self.display_processed_image()
                self.update_histogram()
            except Exception as e:
                print(f"Ошибка при обработке изображения: {e}")
                messagebox.showerror("Ошибка", "Не удалось обработать изображение")

    def rotate_90(self):
        self.processor.rotate_90()
        self.display_processed_image()
        self.update_histogram()
        self.status_var.set("Изображение повернуто на 90°")

    def reset_settings(self):
        """Сброс всех настроек"""
        self.processor.reset_all_adjustments()
        self.brightness_var.set(50)
        self.contrast_var.set(50)
        self.saturation_var.set(50)
        self.linear_a_var.set(1.0)
        self.gamma_var.set(1.0)
        self.brightness_label.config(text="50")
        self.contrast_label.config(text="50")
        self.saturation_label.config(text="50")

        # Сбрасываем grayscale режим если был
        if self.processor.is_grayscale():
            self.processor.convert_to_rgb()
            self.grayscale_button.config(text="В градации серого")

        self.display_images()
        self.update_histogram()
        self.status_var.set("Все настройки сброшены")

    def apply_linear_correction(self, value):
        """Применение линейной коррекции для ЧБ изображений"""
        if self.processor.is_grayscale():
            self.processor.apply_linear_correction(float(value))
            self.display_processed_image()
            self.update_histogram()

    def apply_gamma_correction(self, value):
        """Применение гамма-коррекции для ЧБ изображений"""
        if self.processor.is_grayscale():
            self.processor.apply_nonlinear_correction(float(value))
            self.display_processed_image()
            self.update_histogram()

    def update_histogram(self, event=None):
        if self.processor.processed_image:
            for widget in self.hist_canvas_frame.winfo_children():
                widget.destroy()

            channel = self.hist_channel.get()
            fig = self.processor.create_histogram(channel)

            if fig:
                canvas = FigureCanvasTkAgg(fig, self.hist_canvas_frame)
                canvas.draw()
                canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.N, tk.E, tk.W, tk.S))

    def save_image(self):
        if self.processor.processed_image:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All files", "*.*")]
            )
            if file_path:
                if self.processor.save_image(file_path):
                    self.status_var.set(f"Изображение сохранено как: {os.path.basename(file_path)}")
                    messagebox.showinfo("Выполнено","Изображение успешно сохранено")
                else:
                    messagebox.showerror("Ошибка", "Не удалось сохранить изображение")
        else:
            messagebox.showwarning("Предупреждение", "Нет обработанного изображения для сохранения")