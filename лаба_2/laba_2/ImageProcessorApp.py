import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processor")
        self.root.geometry("1200x800")

        # Переменные для хранения изображений
        self.original_image = None
        self.current_image = None
        self.display_scale = 1.0
        # Для управления оригиналом (цвет/ч\б)
        self._original_color_image = None
        self._original_gray_as_rgb = None
        self._is_original_grayscale = False
        # Стек для отмены изменений текущего изображения
        self._history_stack = []
        # Параметры настройки изображения
        self.brightness = 0  # -100 до 100
        self.contrast = 1.0  # 0.0 до 2.0
        self.saturation = 1.0  # 0.0 до 2.0
        self.hue_shift = 0  # -180 до 180
        self.sharpness = 0  # -100 до 100
        # Переменные для ползунков (для обновления значений)
        self.slider_vars = {}
        self._updating_sliders = False  # Флаг для предотвращения обновлений при программном изменении

        # Применяем темную тему
        self.apply_dark_theme()
        
        self.setup_ui()

    def apply_dark_theme(self):
        """Применение темной темы к приложению"""
        # Цвета темной темы (улучшенная палитра)
        self.dark_bg = "#1e1e1e"  # Темно-серый фон
        self.dark_fg = "#e0e0e0"  # Светло-серый текст (лучше читается)
        self.dark_select_bg = "#3d3d3d"  # Цвет выделения
        self.dark_select_fg = "#ffffff"
        self.dark_entry_bg = "#2d2d2d"  # Фон полей ввода
        self.dark_entry_fg = "#e0e0e0"
        self.dark_button_bg = "#404040"  # Фон кнопок
        self.dark_button_fg = "#e0e0e0"
        self.dark_frame_bg = "#252525"  # Фон фреймов
        self.dark_label_bg = "#1e1e1e"  # Фон меток
        self.dark_canvas_bg = "#2a2a2a"  # Фон для Canvas (немного светлее)
        
        # Настройка основного окна
        self.root.configure(bg=self.dark_bg)
        
        # Настройка ttk.Style для темной темы
        style = ttk.Style()
        style.theme_use('clam')  # Используем 'clam' как базовую тему
        
        # Настройка стилей для различных виджетов
        style.configure('TFrame', background=self.dark_frame_bg)
        style.configure('TLabelFrame', background=self.dark_frame_bg, 
                       foreground=self.dark_fg, borderwidth=1, relief='solid',
                       bordercolor='#404040')
        style.configure('TLabelFrame.Label', background=self.dark_frame_bg, 
                       foreground=self.dark_fg, font=('Arial', 9, 'bold'))
        
        style.configure('TLabel', background=self.dark_label_bg, 
                       foreground=self.dark_fg)
        style.configure('TButton', background=self.dark_button_bg, 
                       foreground=self.dark_fg, borderwidth=1, 
                       focuscolor='none', padding=5)
        style.map('TButton',
                 background=[('active', '#505050'), ('pressed', '#353535'), ('!disabled', self.dark_button_bg)],
                 foreground=[('active', self.dark_fg), ('pressed', self.dark_fg), ('!disabled', self.dark_button_fg)],
                 bordercolor=[('active', '#606060'), ('pressed', '#404040')])
        
        style.configure('TEntry', fieldbackground=self.dark_entry_bg, 
                       foreground=self.dark_entry_fg, borderwidth=1, 
                       insertcolor=self.dark_fg)
        style.map('TEntry',
                 fieldbackground=[('focus', self.dark_entry_bg), ('!focus', self.dark_entry_bg)],
                 bordercolor=[('focus', '#4da6ff'), ('!focus', '#404040')])
        
        style.configure('TScale', background=self.dark_frame_bg, 
                       troughcolor=self.dark_select_bg, 
                       sliderthickness=15, borderwidth=0,
                       darkcolor=self.dark_select_bg,
                       lightcolor=self.dark_select_bg)
        style.map('TScale',
                 background=[('active', '#4da6ff')],
                 slidercolor=[('active', '#4da6ff'), ('!active', '#0078d4')],
                 troughcolor=[('active', self.dark_select_bg)])
        
        style.configure('TSpinbox', fieldbackground=self.dark_entry_bg, 
                       foreground=self.dark_entry_fg, borderwidth=1,
                       insertcolor=self.dark_fg)
        style.map('TSpinbox',
                 fieldbackground=[('focus', self.dark_entry_bg), ('!focus', self.dark_entry_bg)],
                 bordercolor=[('focus', '#4da6ff'), ('!focus', '#404040')])

    def setup_ui(self):
        # Основной фрейм
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Верхняя панель с кнопками загрузки/сохранения
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=5)

        ttk.Button(top_frame, text="Загрузить изображение",
                   command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Сохранить изображение",
                   command=self.save_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Сбросить к оригиналу",
                   command=self.reset_to_original).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Отменить последнее",
                   command=self.undo_last_change).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Ориг. в Ч/Б",
                   command=self.set_original_grayscale).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Ориг. цвет",
                   command=self.set_original_color).pack(side=tk.LEFT, padx=5)

        # Фрейм для отображения изображений
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Оригинальное изображение
        orig_label_frame = ttk.LabelFrame(display_frame, text="Оригинальное изображение")
        orig_label_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.original_canvas = tk.Canvas(orig_label_frame, bg=self.dark_canvas_bg, 
                                         highlightthickness=0, relief='flat')
        # Не растягиваем по всей доступной высоте, чтобы не скрывать панель операций
        self.original_canvas.pack(fill=tk.BOTH, expand=False)
        # Базовый размер, увеличенный по сравнению с исходным, но без чрезмерной высоты
        self.original_canvas.configure(width=560, height=350)

        # Текущее изображение
        current_label_frame = ttk.LabelFrame(display_frame, text="Обработанное изображение")
        current_label_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        self.current_canvas = tk.Canvas(current_label_frame, bg=self.dark_canvas_bg, 
                                        highlightthickness=0, relief='flat')
        self.current_canvas.pack(fill=tk.BOTH, expand=False)
        self.current_canvas.configure(width=560, height=350)

        # Панель операций
        operations_frame = ttk.LabelFrame(main_frame, text="Операции обработки")
        operations_frame.pack(fill=tk.X, pady=10)

        # Морфологические операции
        morph_frame = ttk.Frame(operations_frame)
        morph_frame.pack(fill=tk.X, pady=5)

        ttk.Label(morph_frame, text="Морфологические операции:").pack(anchor=tk.W)

        morph_buttons_frame = ttk.Frame(morph_frame)
        morph_buttons_frame.pack(fill=tk.X, pady=5)

        morph_operations = [
            ("Эрозия", self.apply_erosion),
            ("Дилатация", self.apply_dilation),
            ("Открытие", self.apply_opening),
            ("Закрытие", self.apply_closing),
            ("Градиент", self.apply_gradient),
            ("Top Hat", self.apply_tophat),
            ("Black Hat", self.apply_blackhat)
        ]

        for text, command in morph_operations:
            ttk.Button(morph_buttons_frame, text=text,
                       command=command).pack(side=tk.LEFT, padx=2)

        # Поле для ввода структурного элемента
        kernel_frame = ttk.Frame(morph_frame)
        kernel_frame.pack(fill=tk.X, pady=5)

        ttk.Label(kernel_frame, text="Структурный элемент (матрица через запятую, например: 1,1,1;1,1,1;1,1,1):").pack(
            anchor=tk.W)
        self.kernel_entry = ttk.Entry(kernel_frame, width=50)
        self.kernel_entry.pack(fill=tk.X, pady=2)
        self.kernel_entry.insert(0, "1,1,1;1,1,1;1,1,1")
        ttk.Button(kernel_frame, text="Задать матрицу через таблицу",
                   command=lambda: self.open_matrix_dialog(self.kernel_entry, value_type=float, int_cast=True)).pack(anchor=tk.W, pady=2)

        # Другие операции
        other_ops_frame = ttk.Frame(operations_frame)
        other_ops_frame.pack(fill=tk.X, pady=5)

        ttk.Button(other_ops_frame, text="Повышение резкости",
                   command=self.apply_sharpening).pack(side=tk.LEFT, padx=2)
        ttk.Button(other_ops_frame, text="Размытие в движении",
                   command=self.apply_motion_blur).pack(side=tk.LEFT, padx=2)
        ttk.Button(other_ops_frame, text="Тиснение",
                   command=self.apply_emboss).pack(side=tk.LEFT, padx=2)
        ttk.Button(other_ops_frame, text="Медианная фильтрация",
                   command=self.apply_median_filter).pack(side=tk.LEFT, padx=2)

        # Пользовательский фильтр
        custom_filter_frame = ttk.Frame(operations_frame)
        custom_filter_frame.pack(fill=tk.X, pady=5)

        ttk.Label(custom_filter_frame, text="Пользовательский фильтр (матрица, например: 0,-1,0;-1,5,-1;0,-1,0):").pack(
            anchor=tk.W)
        self.custom_filter_entry = ttk.Entry(custom_filter_frame, width=50)
        self.custom_filter_entry.pack(fill=tk.X, pady=2)
        self.custom_filter_entry.insert(0, "0,-1,0;-1,5,-1;0,-1,0")
        ttk.Button(custom_filter_frame, text="Применить пользовательский фильтр",
                   command=self.apply_custom_filter).pack(pady=2)
        ttk.Button(custom_filter_frame, text="Задать фильтр через таблицу",
                   command=lambda: self.open_matrix_dialog(self.custom_filter_entry, value_type=float)).pack(anchor=tk.W)

        # Панель настройки параметров изображения
        params_frame = ttk.LabelFrame(main_frame, text="🎨 Настройка параметров изображения")
        params_frame.pack(fill=tk.X, pady=10, padx=5)

        # Создаём сетку для ползунков
        params_grid = ttk.Frame(params_frame)
        params_grid.pack(fill=tk.X, padx=10, pady=10)

        # Яркость
        self.slider_vars['brightness'] = self.create_slider(
            params_grid, "Яркость", -100, 100, 0, 
            lambda v: self.update_image_params('brightness', v), row=0)
        
        # Контрастность
        self.slider_vars['contrast'] = self.create_slider(
            params_grid, "Контрастность", 0.0, 2.0, 1.0, 
            lambda v: self.update_image_params('contrast', v), row=1, step=0.01)
        
        # Насыщенность
        self.slider_vars['saturation'] = self.create_slider(
            params_grid, "Насыщенность", 0.0, 2.0, 1.0, 
            lambda v: self.update_image_params('saturation', v), row=2, step=0.01)
        
        # Оттенок
        self.slider_vars['hue'] = self.create_slider(
            params_grid, "Оттенок", -180, 180, 0, 
            lambda v: self.update_image_params('hue', v), row=3)
        
        # Резкость
        self.slider_vars['sharpness'] = self.create_slider(
            params_grid, "Резкость", -100, 100, 0, 
            lambda v: self.update_image_params('sharpness', v), row=4)

        # Кнопка сброса параметров
        reset_params_frame = ttk.Frame(params_frame)
        reset_params_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(reset_params_frame, text="🔄 Сбросить все параметры", 
                  command=self.reset_image_params).pack(side=tk.LEFT, padx=5)

    # ===================== Методы для работы с ползунками =====================
    def create_slider(self, parent, label, min_val, max_val, default_val, callback, row=0, step=1):
        """Создание красивого ползунка с подписями"""
        # Фрейм для одного ползунка
        slider_frame = ttk.Frame(parent)
        slider_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=8, padx=5)
        parent.columnconfigure(0, weight=1)
        
        # Верхняя строка: название и значение
        top_frame = ttk.Frame(slider_frame)
        top_frame.pack(fill=tk.X)
        
        # Название параметра
        name_label = ttk.Label(top_frame, text=f"✨ {label}:", font=('Arial', 10, 'bold'))
        name_label.pack(side=tk.LEFT)
        
        # Текущее значение (справа)
        value_var = tk.StringVar()
        value_label = ttk.Label(top_frame, textvariable=value_var, 
                               font=('Arial', 10), foreground='#4da6ff')
        value_label.pack(side=tk.RIGHT)
        
        # Ползунок
        slider_var = tk.DoubleVar(value=default_val)
        scale = ttk.Scale(slider_frame, from_=min_val, to=max_val, 
                         variable=slider_var, orient=tk.HORIZONTAL,
                         length=400, command=lambda v: self._on_slider_change(
                             v, slider_var, value_var, min_val, max_val, step, callback))
        scale.pack(fill=tk.X, pady=5)
        
        # Нижняя строка: минимальное и максимальное значения
        range_frame = ttk.Frame(slider_frame)
        range_frame.pack(fill=tk.X)
        
        min_label = ttk.Label(range_frame, text=f"{min_val}", 
                             font=('Arial', 8), foreground='#a0a0a0')
        min_label.pack(side=tk.LEFT)
        
        max_label = ttk.Label(range_frame, text=f"{max_val}", 
                             font=('Arial', 8), foreground='#a0a0a0')
        max_label.pack(side=tk.RIGHT)
        
        # Инициализация отображения значения (без вызова callback)
        if step >= 1:
            value_var.set(f"{int(default_val)}")
        else:
            value_var.set(f"{default_val:.2f}")
        
        return slider_var
    
    def _on_slider_change(self, value, slider_var, value_var, min_val, max_val, step, callback):
        """Обработчик изменения ползунка"""
        try:
            val = float(value)
            # Округляем в зависимости от шага
            if step >= 1:
                val = round(val)
            else:
                val = round(val / step) * step
            val = max(min_val, min(max_val, val))
            
            # Обновляем отображение значения
            if step >= 1:
                value_var.set(f"{int(val)}")
            else:
                value_var.set(f"{val:.2f}")
            
            # Вызываем callback
            callback(val)
        except:
            pass
    
    def update_image_params(self, param_name, value):
        """Обновление параметра изображения"""
        if self.current_image is None or self._updating_sliders:
            return
        
        # Сохраняем параметр
        if param_name == 'brightness':
            self.brightness = int(value)
        elif param_name == 'contrast':
            self.contrast = float(value)
        elif param_name == 'saturation':
            self.saturation = float(value)
        elif param_name == 'hue':
            self.hue_shift = int(value)
        elif param_name == 'sharpness':
            self.sharpness = int(value)
        
        # Применяем все параметры к изображению
        self.apply_image_params()
    
    def apply_image_params(self):
        """Применение всех параметров к изображению"""
        if self.current_image is None or self.original_image is None:
            return
        
        try:
            # Начинаем с оригинального изображения
            img = self.original_image.copy()
            
            # Применяем яркость и контрастность
            if self.brightness != 0 or self.contrast != 1.0:
                img = cv2.convertScaleAbs(img, alpha=self.contrast, beta=self.brightness)
            
            # Применяем насыщенность и оттенок (только для цветных изображений)
            if len(img.shape) == 3 and (self.saturation != 1.0 or self.hue_shift != 0):
                # Конвертируем в HSV
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
                
                # Изменяем насыщенность
                if self.saturation != 1.0:
                    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.saturation, 0, 255)
                
                # Изменяем оттенок
                if self.hue_shift != 0:
                    hsv[:, :, 0] = (hsv[:, :, 0] + self.hue_shift) % 180
                
                # Конвертируем обратно в RGB
                hsv = hsv.astype(np.uint8)
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            # Применяем резкость
            if self.sharpness != 0:
                # Создаём ядро для резкости
                intensity = abs(self.sharpness) / 100.0
                if self.sharpness > 0:
                    # Увеличение резкости
                    kernel = np.array([[-1, -1, -1],
                                     [-1, 9 + intensity * 5, -1],
                                     [-1, -1, -1]])
                else:
                    # Размытие
                    kernel = np.ones((5, 5), np.float32) / (25.0 + intensity * 10)
                img = cv2.filter2D(img, -1, kernel)
            
            # Обновляем текущее изображение
            self.current_image = img
            self.update_displays()
            
        except Exception as e:
            print(f"Ошибка применения параметров: {e}")
    
    def reset_image_params_silent(self):
        """Тихий сброс параметров (без сообщения)"""
        # Временно отключаем обновление изображения при изменении ползунков
        self._updating_sliders = True
        
        self.brightness = 0
        self.contrast = 1.0
        self.saturation = 1.0
        self.hue_shift = 0
        self.sharpness = 0
        
        # Обновляем ползунки (это не вызовет callback благодаря флагу)
        if 'brightness' in self.slider_vars:
            self.slider_vars['brightness'].set(0)
        if 'contrast' in self.slider_vars:
            self.slider_vars['contrast'].set(1.0)
        if 'saturation' in self.slider_vars:
            self.slider_vars['saturation'].set(1.0)
        if 'hue' in self.slider_vars:
            self.slider_vars['hue'].set(0)
        if 'sharpness' in self.slider_vars:
            self.slider_vars['sharpness'].set(0)
        
        self._updating_sliders = False
    
    def reset_image_params(self):
        """Сброс всех параметров к значениям по умолчанию"""
        self.reset_image_params_silent()
        
        # Обновляем изображение
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.update_displays()
        
        messagebox.showinfo("Сброс", "Все параметры сброшены к значениям по умолчанию")

    # ===================== Служебные методы состояния =====================
    def push_history(self):
        if self.current_image is not None:
            self._history_stack.append(self.current_image.copy())

    def undo_last_change(self):
        if not self._history_stack:
            messagebox.showinfo("Отмена", "Нет изменений для отмены")
            return
        try:
            self.current_image = self._history_stack.pop()
            self.update_displays()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось выполнить отмену: {e}")

    def set_original_grayscale(self):
        if self._original_gray_as_rgb is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return
        self.original_image = self._original_gray_as_rgb.copy()
        self._is_original_grayscale = True
        self.update_displays()

    def set_original_color(self):
        if self._original_color_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return
        self.original_image = self._original_color_image.copy()
        self._is_original_grayscale = False
        self.update_displays()

    # ===================== Диалог ввода матрицы =====================
    def open_matrix_dialog(self, target_entry, value_type=float, int_cast=False):
        dialog = tk.Toplevel(self.root)
        dialog.title("Ввод матрицы")
        dialog.grab_set()
        # Применяем темную тему к диалогу
        dialog.configure(bg=self.dark_bg)

        size_frame = ttk.Frame(dialog)
        size_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(size_frame, text="Строки:").pack(side=tk.LEFT)
        rows_var = tk.IntVar(value=3)
        ttk.Spinbox(size_frame, from_=1, to=15, textvariable=rows_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(size_frame, text="Столбцы:").pack(side=tk.LEFT, padx=(10, 0))
        cols_var = tk.IntVar(value=3)
        ttk.Spinbox(size_frame, from_=1, to=15, textvariable=cols_var, width=5).pack(side=tk.LEFT, padx=5)

        grid_container = ttk.Frame(dialog)
        grid_container.pack(padx=10, pady=5)

        entries = []

        def build_grid():
            for w in grid_container.winfo_children():
                w.destroy()
            entries.clear()
            r = rows_var.get()
            c = cols_var.get()
            for i in range(r):
                row_entries = []
                for j in range(c):
                    e = ttk.Entry(grid_container, width=6)
                    e.grid(row=i, column=j, padx=2, pady=2)
                    e.insert(0, "1" if int_cast else "0")
                    row_entries.append(e)
                entries.append(row_entries)

        build_grid()

        controls = ttk.Frame(dialog)
        controls.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(controls, text="Обновить размер", command=build_grid).pack(side=tk.LEFT)

        def apply_matrix():
            try:
                matrix = []
                for row in entries:
                    values = []
                    for e in row:
                        val = value_type(e.get().strip())
                        if int_cast:
                            val = int(round(val))
                        values.append(val)
                    matrix.append(values)
                # Сериализуем в строку для существующего парсера
                serialized = ";".join([",".join(str(x) for x in row) for row in matrix])
                target_entry.delete(0, tk.END)
                target_entry.insert(0, serialized)
                dialog.destroy()
            except Exception as ex:
                messagebox.showerror("Ошибка", f"Некорректное значение матрицы: {ex}")

        ttk.Button(controls, text="Применить", command=apply_matrix).pack(side=tk.RIGHT)
        ttk.Button(controls, text="Отмена", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

    def parse_kernel(self, kernel_str):
        """Парсинг структурного элемента из строки"""
        try:
            rows = kernel_str.split(';')
            kernel = []
            for row in rows:
                kernel.append([int(x.strip()) for x in row.split(',')])
            return np.array(kernel, dtype=np.uint8)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Неправильный формат матрицы: {e}")
            return None

    def parse_custom_kernel(self, kernel_str):
        """Парсинг пользовательского фильтра из строки"""
        try:
            rows = kernel_str.split(';')
            kernel = []
            for row in rows:
                kernel.append([float(x.strip()) for x in row.split(',')])
            return np.array(kernel)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Неправильный формат матрицы: {e}")
            return None

    def load_image(self):
        """Загрузка изображения только через PIL"""
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("Bitmap", "*.bmp"),
                ("TIFF", "*.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        if not file_path:
            return

        try:
            # Загружаем изображение через PIL
            pil_image = Image.open(file_path)

            # Конвертируем в RGB если необходимо
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # Конвертируем в numpy array для обработки в OpenCV
            numpy_image = np.array(pil_image)

            # Сохраняем изображение
            self._original_color_image = numpy_image
            gray = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)
            self._original_gray_as_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            self.original_image = self._original_color_image.copy()
            self._is_original_grayscale = False
            self.current_image = self.original_image.copy()
            self._history_stack = []
            # Сбрасываем параметры при загрузке нового изображения
            self.reset_image_params_silent()
            self.update_displays()
            messagebox.showinfo("Успех", "Изображение успешно загружено")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка загрузки изображения: {str(e)}")

    def save_image(self):
        """Сохранение изображения только через PIL"""
        if self.current_image is not None:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("Bitmap files", "*.bmp"),
                    ("TIFF files", "*.tiff"),
                    ("All files", "*.*")
                ]
            )
            if file_path:
                try:
                    # Конвертируем numpy array обратно в PIL Image
                    # Убеждаемся, что тип данных правильный
                    if self.current_image.dtype != np.uint8:
                        image_to_save = Image.fromarray(self.current_image.astype(np.uint8))
                    else:
                        image_to_save = Image.fromarray(self.current_image)

                    # Сохраняем через PIL
                    image_to_save.save(file_path)
                    messagebox.showinfo("Успех", f"Изображение сохранено: {file_path}")

                except Exception as e:
                    messagebox.showerror("Ошибка", f"Ошибка сохранения: {str(e)}")
        else:
            messagebox.showwarning("Предупреждение", "Нет изображения для сохранения")

    def reset_to_original(self):
        """Сброс к оригинальному изображению"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.update_displays()

    def update_displays(self):
        """Обновление отображения изображений"""
        if self.original_image is not None:
            self.display_image(self.original_image, self.original_canvas)
        if self.current_image is not None:
            self.display_image(self.current_image, self.current_canvas)

    def display_image(self, image, canvas):
        """Отображение изображения на canvas"""
        try:
            # Масштабируем изображение под фактический размер canvas (включая увеличение)
            h, w = image.shape[:2]
            max_w = canvas.winfo_width()
            max_h = canvas.winfo_height()
            if max_w <= 1 or max_h <= 1:
                # Если canvas ещё не отрисован, используем базовые значения
                max_w, max_h = 560, 720

            scale = min(max_h / h, max_w / w)
            new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
            # Выбор интерполяции: AREA для уменьшения, CUBIC для увеличения
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            display_image = cv2.resize(image, (new_w, new_h), interpolation=interp)

            # Конвертируем для Tkinter
            pil_image = Image.fromarray(display_image)
            tk_image = ImageTk.PhotoImage(pil_image)

            # Обновляем canvas
            canvas.delete("all")
            canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
            canvas.image = tk_image  # Сохраняем ссылку
        except Exception as e:
            print(f"Ошибка отображения изображения: {e}")

    def apply_morphological_operation(self, operation):
        """Применение морфологической операции"""
        if self.current_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return

        kernel_str = self.kernel_entry.get()
        kernel = self.parse_kernel(kernel_str)
        if kernel is None:
            return

        try:
            self.push_history()
            # Конвертируем в grayscale для морфологических операций
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = self.current_image

            result = operation(gray, kernel)

            # Если было цветное, конвертируем обратно
            if len(self.current_image.shape) == 3:
                self.current_image = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
            else:
                self.current_image = result

            self.update_displays()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка применения операции: {e}")

    def apply_erosion(self):
        self.apply_morphological_operation(cv2.erode)

    def apply_dilation(self):
        self.apply_morphological_operation(cv2.dilate)

    def apply_opening(self):
        self.apply_morphological_operation(
            lambda img, kernel: cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        )

    def apply_closing(self):
        self.apply_morphological_operation(
            lambda img, kernel: cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        )

    def apply_gradient(self):
        self.apply_morphological_operation(
            lambda img, kernel: cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        )

    def apply_tophat(self):
        self.apply_morphological_operation(
            lambda img, kernel: cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        )

    def apply_blackhat(self):
        self.apply_morphological_operation(
            lambda img, kernel: cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        )

    def apply_sharpening(self):
        """Повышение резкости"""
        if self.current_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return

        try:
            self.push_history()
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
            self.current_image = cv2.filter2D(self.current_image, -1, kernel)
            self.update_displays()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка повышения резкости: {e}")

    def apply_motion_blur(self):
        """Размытие в движении"""
        if self.current_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return

        try:
            self.push_history()
            size = 15
            kernel = np.zeros((size, size))
            kernel[int((size - 1) / 2), :] = np.ones(size)
            kernel = kernel / size
            self.current_image = cv2.filter2D(self.current_image, -1, kernel)
            self.update_displays()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка применения размытия: {e}")

    def apply_emboss(self):
        """Эффект тиснения"""
        if self.current_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return

        try:
            self.push_history()
            kernel = np.array([[-2, -1, 0],
                               [-1, 1, 1],
                               [0, 1, 2]])
            self.current_image = cv2.filter2D(self.current_image, -1, kernel)
            self.update_displays()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка применения тиснения: {e}")

    def apply_median_filter(self):
        """Медианная фильтрация"""
        if self.current_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return

        try:
            self.push_history()
            if len(self.current_image.shape) == 3:
                self.current_image = cv2.medianBlur(self.current_image, 5)
            else:
                self.current_image = cv2.medianBlur(self.current_image, 5)
            self.update_displays()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка медианной фильтрации: {e}")

    def apply_custom_filter(self):
        """Применение пользовательского фильтра"""
        if self.current_image is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            return

        kernel_str = self.custom_filter_entry.get()
        kernel = self.parse_custom_kernel(kernel_str)
        if kernel is None:
            return

        try:
            self.push_history()
            self.current_image = cv2.filter2D(self.current_image, -1, kernel)
            self.update_displays()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка применения фильтра: {e}")
