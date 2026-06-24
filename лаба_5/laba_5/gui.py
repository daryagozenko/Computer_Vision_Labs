"""
Улучшенный графический интерфейс для управления базой лиц и распознавания.
"""

import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from PIL import Image, ImageTk

import cv2
import face_recognition
import numpy as np

import main as core


class FaceApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Система распознавания лиц")
        self.root.geometry("1000x800")
        
        self.status_var = tk.StringVar(value="Готово.")
        self.tolerance_var = tk.DoubleVar(value=0.45)
        
        self.cam_thread: threading.Thread | None = None
        self.video_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.cap: cv2.VideoCapture | None = None
        self.video_updating = False
        
        self._build_ui()
        core._ensure_db()
        self._refresh_profiles()

    def _build_ui(self) -> None:
        # Создаём notebook для вкладок
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Вкладка 1: Управление базой лиц
        tab_db = ttk.Frame(notebook)
        notebook.add(tab_db, text="База лиц")
        self._build_database_tab(tab_db)
        
        # Вкладка 2: Распознавание
        tab_rec = ttk.Frame(notebook)
        notebook.add(tab_rec, text="Распознавание")
        self._build_recognition_tab(tab_rec)
        
        # Вкладка 3: Настройки
        tab_settings = ttk.Frame(notebook)
        notebook.add(tab_settings, text="Настройки")
        self._build_settings_tab(tab_settings)
        
        # Статусная строка внизу
        status_frame = tk.Frame(self.root)
        status_frame.pack(fill="x", side="bottom", padx=5, pady=5)
        tk.Label(status_frame, textvariable=self.status_var, anchor="w", relief="sunken", bd=1).pack(fill="x")

    def _build_database_tab(self, parent: ttk.Frame) -> None:
        # Верхняя панель с кнопками
        top_frame = tk.Frame(parent)
        top_frame.pack(fill="x", padx=10, pady=10)
        
        tk.Button(top_frame, text="➕ Добавить лицо", command=self.add_person, 
                 bg="#4CAF50", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        tk.Button(top_frame, text="🔄 Обновить список", command=self._refresh_profiles,
                 bg="#2196F3", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=5)
        
        # Canvas с прокруткой для списка профилей
        canvas_frame = tk.Frame(parent)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        scrollbar = tk.Scrollbar(canvas_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.profiles_canvas = tk.Canvas(canvas_frame, yscrollcommand=scrollbar.set)
        self.profiles_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.profiles_canvas.yview)
        
        self.profiles_frame = tk.Frame(self.profiles_canvas)
        self.profiles_canvas.create_window((0, 0), window=self.profiles_frame, anchor="nw")
        
        self.profiles_frame.bind("<Configure>", lambda e: self.profiles_canvas.configure(scrollregion=self.profiles_canvas.bbox("all")))

    def _build_recognition_tab(self, parent: ttk.Frame) -> None:
        # Группа для распознавания по изображению
        img_frame = tk.LabelFrame(parent, text="Распознавание по изображению", font=("Arial", 10, "bold"))
        img_frame.pack(fill="x", padx=10, pady=10)
        
        tk.Button(img_frame, text="📷 Выбрать изображение", command=self.recognize_image,
                 bg="#FF9800", fg="white", font=("Arial", 10, "bold"), width=30).pack(pady=10)
        
        self.recognition_result_label = tk.Label(img_frame, text="", font=("Arial", 12), fg="#1976D2")
        self.recognition_result_label.pack(pady=5)
        
        # Группа для распознавания в реальном времени
        realtime_frame = tk.LabelFrame(parent, text="Распознавание в реальном времени", font=("Arial", 10, "bold"))
        realtime_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Используем grid для всей структуры
        realtime_frame.grid_rowconfigure(0, weight=0)  # Кнопки - фиксированный размер
        realtime_frame.grid_rowconfigure(1, weight=0)  # Подпись с именем - фиксированный размер
        realtime_frame.grid_rowconfigure(2, weight=0)  # Статус камеры - фиксированный размер
        realtime_frame.grid_rowconfigure(3, weight=1, minsize=550)  # Видео - растягивается
        realtime_frame.grid_columnconfigure(0, weight=1)
        
        # Кнопки управления
        buttons_frame = tk.Frame(realtime_frame)
        buttons_frame.grid(row=0, column=0, sticky="ew", pady=10, padx=10)
        
        self.camera_btn = tk.Button(buttons_frame, text="📹 Запустить с камеры", command=self.start_camera,
                                   bg="#9C27B0", fg="white", font=("Arial", 10, "bold"), width=20)
        self.camera_btn.pack(side="left", padx=5)
        
        self.video_btn = tk.Button(buttons_frame, text="🎬 Запустить с видео", command=self.start_video,
                                   bg="#E91E63", fg="white", font=("Arial", 10, "bold"), width=20)
        self.video_btn.pack(side="left", padx=5)
        
        self.stop_btn = tk.Button(buttons_frame, text="⏹ Остановить", command=self.stop_recognition,
                                  bg="#F44336", fg="white", font=("Arial", 10, "bold"), width=20)
        self.stop_btn.pack(side="left", padx=5)
        
        # Подпись с последним распознанным именем - ПРЯМО ПОД КНОПКАМИ, всегда видна
        self.live_name_label = tk.Label(
            realtime_frame,
            text="Последнее распознанное: -",
            font=("Arial", 14, "bold"),
            fg="#FFD54F",
            bg="#1976D2",
            anchor="w",
            padx=15,
            pady=12,
            relief="solid",
            bd=2,
        )
        self.live_name_label.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))
        
        # Информация о доступности камеры
        self.camera_status_label = tk.Label(realtime_frame, text="", font=("Arial", 9))
        self.camera_status_label.grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self._update_camera_status()
        
        # Контейнер для видео
        video_container = tk.Frame(realtime_frame, bg="black")
        video_container.grid(row=3, column=0, sticky="nsew", padx=10, pady=10)
        video_container.grid_columnconfigure(0, weight=1)
        video_container.grid_rowconfigure(0, weight=1)

        # Label для отображения видео
        self.video_label = tk.Label(
            video_container,
            text="Видео будет отображаться здесь",
            bg="black",
            fg="white",
        )
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        # Переменная для хранения текущего кадра
        self.current_frame = None

    def _build_settings_tab(self, parent: ttk.Frame) -> None:
        settings_frame = tk.Frame(parent)
        settings_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Настройка порога распознавания
        tolerance_frame = tk.LabelFrame(settings_frame, text="Порог распознавания (tolerance)", font=("Arial", 10, "bold"))
        tolerance_frame.pack(fill="x", pady=10)
        
        tk.Label(tolerance_frame, text="Чем меньше значение, тем строже распознавание (рекомендуется: 0.4-0.6)", 
                font=("Arial", 9)).pack(pady=5)
        
        scale_frame = tk.Frame(tolerance_frame)
        scale_frame.pack(fill="x", padx=10, pady=10)
        
        self.tolerance_scale = tk.Scale(scale_frame, from_=0.1, to=1.0, resolution=0.05,
                                        orient="horizontal", variable=self.tolerance_var,
                                        command=self._on_tolerance_change, length=400)
        self.tolerance_scale.pack(side="left")
        
        self.tolerance_label = tk.Label(scale_frame, text="0.45", font=("Arial", 10, "bold"), width=6)
        self.tolerance_label.pack(side="left", padx=10)
        
        # Информация о базе данных
        db_frame = tk.LabelFrame(settings_frame, text="Информация о базе данных", font=("Arial", 10, "bold"))
        db_frame.pack(fill="x", pady=10)
        
        self.db_info_label = tk.Label(db_frame, text="", font=("Arial", 9), justify="left")
        self.db_info_label.pack(padx=10, pady=10)
        self._update_db_info()

    def _on_tolerance_change(self, value: str) -> None:
        self.tolerance_label.config(text=f"{float(value):.2f}")

    def _update_camera_status(self) -> None:
        if core.check_camera_available():
            self.camera_status_label.config(text="✅ Веб-камера доступна", fg="green")
        else:
            self.camera_status_label.config(text="❌ Веб-камера недоступна. Используйте видеофайл.", fg="red")

    def _update_db_info(self) -> None:
        try:
            persons = core.list_persons()
            count = len(persons)
            self.db_info_label.config(text=f"Всего профилей в базе: {count}\nПуть к базе: {core.DB_PATH}")
        except Exception:
            self.db_info_label.config(text="Ошибка загрузки информации о базе")

    def _refresh_profiles(self) -> None:
        # Очищаем текущий список
        for widget in self.profiles_frame.winfo_children():
            widget.destroy()
        
        try:
            persons = core.list_persons()
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось загрузить профили: {exc}")
            return
        
        if not persons:
            tk.Label(self.profiles_frame, text="База пуста. Добавьте первый профиль.", 
                    font=("Arial", 12), fg="gray").pack(pady=20)
            return
        
        # Отображаем каждый профиль
        for pid, name, path in persons:
            self._add_profile_widget(pid, name, path)
        
        self._update_db_info()

    def _add_profile_widget(self, pid: int, name: str, path: str) -> None:
        profile_frame = tk.Frame(self.profiles_frame, relief="ridge", bd=2, bg="white")
        profile_frame.pack(fill="x", padx=5, pady=5)
        
        # Миниатюра изображения
        img_frame = tk.Frame(profile_frame, bg="white")
        img_frame.pack(side="left", padx=10, pady=10)
        
        try:
            if path and os.path.isfile(path):
                img = Image.open(path)
                img.thumbnail((80, 80), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                img_label = tk.Label(img_frame, image=photo, bg="white")
                img_label.image = photo  # Сохраняем ссылку
                img_label.pack()
            else:
                tk.Label(img_frame, text="Нет\nизображения", bg="lightgray", width=10, height=4).pack()
        except Exception:
            tk.Label(img_frame, text="Ошибка\nзагрузки", bg="lightgray", width=10, height=4).pack()
        
        # Информация о профиле
        info_frame = tk.Frame(profile_frame, bg="white")
        info_frame.pack(side="left", fill="both", expand=True, padx=10)
        
        tk.Label(info_frame, text=f"ID: {pid}", font=("Arial", 9, "bold"), bg="white").pack(anchor="w")
        tk.Label(info_frame, text=f"Имя: {name}", font=("Arial", 10), bg="white").pack(anchor="w")
        if path:
            tk.Label(info_frame, text=f"Путь: {os.path.basename(path)}", font=("Arial", 8), 
                    fg="gray", bg="white").pack(anchor="w")
        
        # Кнопка удаления
        tk.Button(profile_frame, text="🗑 Удалить", command=lambda p=pid: self.delete_person(p),
                 bg="#F44336", fg="white", font=("Arial", 8)).pack(side="right", padx=10)

    def add_person(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp"), ("Все файлы", "*.*")]
        )
        if not path:
            return
        
        # Валидация файла
        if not os.path.isfile(path):
            messagebox.showerror("Ошибка", "Файл не найден")
            return
        
        name = simpledialog.askstring("Имя", "Введите имя для профиля:")
        if not name or not name.strip():
            return
        
        try:
            self.status_var.set("Обработка изображения...")
            self.root.update()
            
            pid = core.add_person(name.strip(), path)
            messagebox.showinfo("Успех", f"Профиль '{name}' успешно добавлен (ID: {pid})")
            self._refresh_profiles()
            self.status_var.set("Профиль добавлен")
        except ValueError as exc:
            messagebox.showerror("Ошибка валидации", str(exc))
            self.status_var.set("Ошибка валидации")
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось добавить профиль: {exc}")
            self.status_var.set("Ошибка добавления профиля")

    def delete_person(self, person_id: int) -> None:
        if not messagebox.askyesno("Подтверждение", f"Удалить профиль с ID {person_id}?"):
            return
        
        try:
            if core.delete_person(person_id):
                messagebox.showinfo("Успех", "Профиль удалён")
                self._refresh_profiles()
                self.status_var.set("Профиль удалён")
            else:
                messagebox.showerror("Ошибка", "Профиль не найден")
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось удалить профиль: {exc}")

    def recognize_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите изображение для распознавания",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp"), ("Все файлы", "*.*")]
        )
        if not path:
            return
        
        try:
            self.status_var.set("Распознавание...")
            self.root.update()
            
            tolerance = self.tolerance_var.get()
            name = core.recognize_from_image(path, tolerance=tolerance)
            
            if name:
                self.recognition_result_label.config(text=f"✅ Распознано: {name}", fg="green")
                self.status_var.set(f"Распознано: {name}")
            else:
                self.recognition_result_label.config(text="❌ Совпадений не найдено", fg="red")
                self.status_var.set("Совпадений не найдено")
        except RuntimeError as exc:
            messagebox.showerror("Ошибка", str(exc))
            self.recognition_result_label.config(text="", fg="black")
            self.status_var.set("Ошибка распознавания")
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось распознать лицо: {exc}")
            self.recognition_result_label.config(text="", fg="black")
            self.status_var.set("Ошибка распознавания")

    def start_camera(self) -> None:
        if not core.check_camera_available():
            messagebox.showerror("Ошибка", "Веб-камера недоступна. Используйте распознавание с видеофайла.")
            return
        
        if (self.cam_thread and self.cam_thread.is_alive()) or (self.video_thread and self.video_thread.is_alive()):
            messagebox.showinfo("Внимание", "Распознавание уже запущено. Остановите его сначала.")
            return
        
        try:
            encodings, names = core._load_all_encodings()
            if not encodings:
                messagebox.showwarning("База пуста", "Добавьте хотя бы один профиль в базу.")
                return
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось загрузить базу: {exc}")
            return
        
        self.stop_event.clear()
        self.cam_thread = threading.Thread(target=self._camera_worker, daemon=True)
        self.cam_thread.start()
        self.status_var.set("Распознавание с камеры запущено (нажмите 'q' в окне для выхода)")

    def start_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите видеофайл",
            filetypes=[("Видео", "*.mp4 *.avi *.mov *.mkv"), ("Все файлы", "*.*")]
        )
        if not path:
            return
        
        if (self.cam_thread and self.cam_thread.is_alive()) or (self.video_thread and self.video_thread.is_alive()):
            messagebox.showinfo("Внимание", "Распознавание уже запущено. Остановите его сначала.")
            return
        
        try:
            encodings, names = core._load_all_encodings()
            if not encodings:
                messagebox.showwarning("База пуста", "Добавьте хотя бы один профиль в базу.")
                return
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось загрузить базу: {exc}")
            return
        
        self.stop_event.clear()
        self.video_thread = threading.Thread(target=self._video_worker, args=(path,), daemon=True)
        self.video_thread.start()
        self.status_var.set(f"Распознавание из видео запущено (нажмите 'q' в окне для выхода)")

    def stop_recognition(self) -> None:
        self.stop_event.set()
        self.video_updating = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.cam_thread and self.cam_thread.is_alive():
            self.cam_thread.join(timeout=2)
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=2)
        self.current_frame = None
        self.video_label.config(image="", text="Видео остановлено", bg="lightgray")
        self.status_var.set("Распознавание остановлено")

    def _update_video_frame(self) -> None:
        """Обновляет отображение видео в GUI."""
        if not self.video_updating:
            return
            
        if self.current_frame is not None:
            # Изменяем размер кадра для отображения
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Получаем размеры контейнера видео (более надёжно, чем Label)
            video_container = self.video_label.master
            video_container.update_idletasks()
            self.root.update_idletasks()
            
            container_width = video_container.winfo_width()
            container_height = video_container.winfo_height()
            
            # Используем размеры контейнера напрямую (подпись теперь вне контейнера)
            label_width = container_width if container_width > 1 else 800
            label_height = container_height if container_height > 1 else 550
            
            # Если размеры ещё не определены, используем стандартный размер
            # Это предотвратит постепенное раскрытие
            if label_width <= 1 or label_height <= 1:
                label_width = 800
                label_height = 550
            
            # Вычисляем пропорции для масштабирования с сохранением соотношения сторон
            img_width, img_height = img.size
            scale_w = label_width / img_width
            scale_h = label_height / img_height
            scale = min(scale_w, scale_h)  # Используем меньший масштаб для сохранения пропорций
            
            # Масштабируем изображение
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            if new_width > 0 and new_height > 0:
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo  # Сохраняем ссылку
        
        # Планируем следующее обновление, если распознавание ещё активно
        if self.video_updating and not self.stop_event.is_set():
            self.root.after(30, self._update_video_frame)  # ~30 FPS
    
    def _camera_worker(self) -> None:
        try:
            encodings, names = core._load_all_encodings()
            tolerance = self.tolerance_var.get()
            
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.root.after(0, lambda: messagebox.showerror("Камера", "Не удалось открыть веб-камеру."))
                return
            
            # Запускаем обновление GUI
            self.video_updating = True
            self.root.after(0, self._update_video_frame)
            
            try:
                while not self.stop_event.is_set():
                    ok, frame = self.cap.read()
                    if not ok:
                        break
                    
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    locations = face_recognition.face_locations(rgb)
                    frame_encodings = face_recognition.face_encodings(rgb, locations)
                    
                    for (top, right, bottom, left), fe in zip(locations, frame_encodings):
                        distances = face_recognition.face_distance(encodings, fe)
                        name = "Неизвестно"
                        color = (0, 0, 255)  # Красный для неизвестных

                        if len(distances) > 0:
                            idx = np.argmin(distances)
                            if distances[idx] <= tolerance:
                                name = names[idx]
                                color = (0, 255, 0)  # Зелёный для распознанных

                        # Обновляем подпись в GUI (поддерживает кириллицу)
                        display_name = name
                        self.root.after(
                            0,
                            lambda n=display_name: self.live_name_label.config(
                                text=f"Последнее распознанное: {n}"
                            ),
                        )

                        # Текст непосредственно на кадре ограничиваем ASCII,
                        # чтобы избежать вопросительных знаков в OpenCV
                        try:
                            ascii_name = display_name.encode("ascii", "ignore").decode("ascii")
                        except Exception:
                            ascii_name = ""
                        if not ascii_name:
                            ascii_name = "UNKNOWN"

                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        cv2.putText(
                            frame,
                            ascii_name,
                            (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )
                    
                    # Сохраняем кадр для отображения в GUI
                    self.current_frame = frame.copy()
            finally:
                if self.cap:
                    self.cap.release()
                    self.cap = None
                self.current_frame = None
                self.video_updating = False
                self.root.after(0, lambda: self.video_label.config(image="", text="Видео остановлено", bg="lightgray"))
        finally:
            self.stop_event.set()

    def _video_worker(self, video_path: str) -> None:
        try:
            encodings, names = core._load_all_encodings()
            tolerance = self.tolerance_var.get()
            
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                self.root.after(0, lambda: messagebox.showerror("Видео", f"Не удалось открыть видеофайл: {video_path}"))
                return
            
            # Запускаем обновление GUI
            self.video_updating = True
            self.root.after(0, self._update_video_frame)
            
            # Получаем FPS видео для правильной скорости воспроизведения
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30
            delay = int(1000 / fps)  # миллисекунды между кадрами
            
            try:
                while not self.stop_event.is_set():
                    ok, frame = self.cap.read()
                    if not ok:
                        self.root.after(0, lambda: self.video_label.config(image="", text="Видео закончилось", bg="lightgray"))
                        break
                    
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    locations = face_recognition.face_locations(rgb)
                    frame_encodings = face_recognition.face_encodings(rgb, locations)
                    
                    for (top, right, bottom, left), fe in zip(locations, frame_encodings):
                        distances = face_recognition.face_distance(encodings, fe)
                        name = "Неизвестно"
                        color = (0, 0, 255)

                        if len(distances) > 0:
                            idx = np.argmin(distances)
                            if distances[idx] <= tolerance:
                                name = names[idx]
                                color = (0, 255, 0)

                        # Обновляем подпись в GUI (поддерживает кириллицу)
                        display_name = name
                        self.root.after(
                            0,
                            lambda n=display_name: self.live_name_label.config(
                                text=f"Последнее распознанное: {n}"
                            ),
                        )

                        # На самом кадре используем только ASCII
                        try:
                            ascii_name = display_name.encode("ascii", "ignore").decode("ascii")
                        except Exception:
                            ascii_name = ""
                        if not ascii_name:
                            ascii_name = "UNKNOWN"

                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        cv2.putText(
                            frame,
                            ascii_name,
                            (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )
                    
                    # Сохраняем кадр для отображения в GUI
                    self.current_frame = frame.copy()
                    
                    # Задержка для синхронизации с FPS видео
                    time.sleep(1.0 / fps)
            finally:
                if self.cap:
                    self.cap.release()
                    self.cap = None
                self.current_frame = None
                self.video_updating = False
                self.root.after(0, lambda: self.video_label.config(image="", text="Видео остановлено", bg="lightgray"))
        finally:
            self.stop_event.set()


def main() -> None:
    root = tk.Tk()
    app = FaceApp(root)
    
    def on_closing():
        app.stop_recognition()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
