import sys
import os
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from datetime import datetime
import json
import traceback

# PyQt5 импорты
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# Импортируем вашу модель
from models import AircraftClassifier


class ModelLoader:
    """Класс для загрузки моделей"""

    @staticmethod
    def load_model(checkpoint_path):
        """Загружает модель из чекпоинта"""
        print(f"Загрузка модели: {checkpoint_path}")

        try:
            # Загружаем чекпоинт
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"Ключи в чекпоинте: {list(checkpoint.keys())}")

            # Дополнительная информация из чекпоинта
            additional_info = checkpoint.get('additional_info', {}) or {}
            classes_from_checkpoint = additional_info.get('classes')

            # Определяем количество классов
            num_classes = ModelLoader._detect_num_classes(checkpoint)
            print(f"Определено классов: {num_classes}")

            # Определяем архитектуру модели
            model_name = ModelLoader._detect_model_name(checkpoint)
            print(f"Архитектура модели: {model_name}")

            # Создаем модель
            model = AircraftClassifier(
                num_classes=num_classes,
                model_name=model_name,
                pretrained=False,
                fine_tune_mode='all'
            )

            # Загружаем веса
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # Получаем имена классов:
            #   1) приоритетно — из чекпоинта (то, на чём реально обучалась модель)
            #   2) если их нет — из файла / захардкоженного списка как запасной вариант
            if isinstance(classes_from_checkpoint, list) and len(classes_from_checkpoint) == num_classes:
                class_names = classes_from_checkpoint
                print(f"✓ Имена классов загружены из чекпоинта: {len(class_names)}")
            else:
                class_names = ModelLoader._load_class_names(num_classes, checkpoint_path)

            # Информация о модели
            model_info = {
                'path': checkpoint_path,
                'name': os.path.basename(checkpoint_path),
                'num_classes': num_classes,
                'architecture': model_name,
                'accuracy': checkpoint.get('val_acc', 'N/A'),
                'epoch': checkpoint.get('epoch', 'N/A'),
                'timestamp': checkpoint.get('timestamp', 'N/A')
            }

            print(f"✓ Модель успешно загружена")
            return model, class_names, model_info

        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            traceback.print_exc()
            raise

    @staticmethod
    def _detect_num_classes(checkpoint):
        """Определяет количество классов из чекпоинта"""
        state_dict = checkpoint['model_state_dict']

        # Пробуем разные методы определения
        methods = [
            # 1. Из additional_info: список классов сохраняется в train.py через CheckpointManager
            lambda: len(checkpoint.get('additional_info', {}).get('classes'))
            if isinstance(checkpoint.get('additional_info', {}).get('classes'), list)
            else None,

            # 2. По последнему слою классификатора
            lambda: ModelLoader._find_num_classes_from_state_dict(state_dict),

            # 3. Из имени модели в чекпоинте
            lambda: ModelLoader._extract_num_from_name(checkpoint.get('model_name', ''))
        ]

        for method in methods:
            result = method()
            if result is not None:
                return result

        # По умолчанию 20 (из вашей ошибки)
        return 20

    @staticmethod
    def _find_num_classes_from_state_dict(state_dict):
        """Ищет количество классов в state_dict"""
        # Ищем последние слои классификатора
        target_layers = ['fc.9', 'fc.10', 'fc.11', 'fc.12', 'classifier']

        for layer in target_layers:
            weight_key = f'backbone.{layer}.weight'
            bias_key = f'backbone.{layer}.bias'

            if weight_key in state_dict:
                shape = state_dict[weight_key].shape
                if len(shape) == 2:
                    return shape[0]
            elif bias_key in state_dict:
                shape = state_dict[bias_key].shape
                if len(shape) == 1:
                    return shape[0]

        return None

    @staticmethod
    def _extract_num_from_name(name):
        """Извлекает число из имени модели"""
        import re
        if not name:
            return None

        # Ищем паттерн "_число" в конце имени
        match = re.search(r'_(\d+)$', name)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _detect_model_name(checkpoint):
        """Определяет имя модели"""
        model_name = checkpoint.get('model_name', 'resnet50')

        # Нормализуем имя
        if 'resnet' in str(model_name).lower():
            return 'resnet50'
        elif 'efficient' in str(model_name).lower():
            return 'efficientnet'
        elif 'mobilenet' in str(model_name).lower():
            return 'mobilenet'
        elif 'simple' in str(model_name).lower():
            return 'simple_cnn'
        else:
            return 'resnet50'

    @staticmethod
    def _load_class_names(num_classes, checkpoint_path):
        """Загружает или создает имена классов"""
        # Пробуем загрузить из файла
        possible_paths = [
            'class_names.txt',
            'classes.txt',
            'labels.txt',
            os.path.join(os.path.dirname(checkpoint_path), 'class_names.txt'),
            os.path.join(os.path.dirname(checkpoint_path), '..', 'class_names.txt'),
        ]

        for file_path in possible_paths:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        names = [line.strip() for line in f if line.strip()]
                        if len(names) >= num_classes:
                            print(f"✓ Загружены имена классов из {file_path}")
                            return names[:num_classes]
                except Exception as e:
                    print(f"Ошибка загрузки {file_path}: {e}")

        # Создаем имена классов
        print(f"Создаем имена классов для {num_classes} классов")

        # Для 20 классов (предположительно самолеты)
        if num_classes == 20:
            return [
                'Airbus A320',
        'Su 80',
        'Boeing 737-700',
        'Tupolev Tu-160',
        'Boeing 777-200',
        'MiG-35',
        'Embraer E-Jet E-195',
        'Airbus A330-300',
        'Boeing 787 Dreamliner',
        'MC-21-310',
        'Bombardier CRJ-200',
        'Embraer ERJ-145',
        'Airbus A380 XWB',
        'Sukhoi Superjet 100',
        'ATR 42-200',
        'Bombardier Dash 8 Q300',
        'Yakovlev Yak-42',
        'Tupolev Tu-134',
        'Beechcraft 1900',
        'Airbus A350 XWB'
            ]
        else:
            return [f"Класс {i + 1}" for i in range(num_classes)]


class ImageProcessor:
    """Класс для обработки изображений"""

    @staticmethod
    def get_transform():
        """Возвращает трансформации для изображения"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @staticmethod
    def predict_image(model, image_path, class_names):
        """Делает предсказание для изображения"""
        try:
            # Загружаем и преобразуем изображение
            image = Image.open(image_path).convert('RGB')
            transform = ImageProcessor.get_transform()
            input_tensor = transform(image).unsqueeze(0)

            # Предсказание
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

            # Получаем топ-N результатов
            top_n = min(5, len(class_names))
            top_probs, top_indices = torch.topk(probabilities, top_n)

            # Формируем результаты
            predictions = []
            for i in range(top_n):
                class_idx = top_indices[i].item()
                confidence = top_probs[i].item() * 100
                predictions.append({
                    'class': class_names[class_idx],
                    'confidence': confidence,
                    'index': class_idx
                })

            return {
                'success': True,
                'image': image,
                'predictions': predictions,
                'best_prediction': predictions[0],
                'image_size': image.size
            }

        except Exception as e:
            print(f"Ошибка обработки изображения: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }


class MainWindow(QMainWindow):
    """Главное окно приложения"""

    def __init__(self):
        super().__init__()
        self.model = None
        self.class_names = []
        self.model_info = {}
        self.current_image_path = None

        self.init_ui()
        self.load_available_models()

    def init_ui(self):
        """Инициализация интерфейса"""
        self.setWindowTitle("Aircraft Classifier - Тестирование модели")
        self.setGeometry(100, 100, 1400, 800)

        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Левая панель - управление
        left_panel = QVBoxLayout()

        # 1. Выбор модели
        model_group = QGroupBox("Модель")
        model_layout = QVBoxLayout()

        self.model_combo = QComboBox()
        self.model_combo.setMinimumHeight(30)
        model_layout.addWidget(QLabel("Выберите модель:"))
        model_layout.addWidget(self.model_combo)

        self.load_model_btn = QPushButton("Загрузить модель")
        self.load_model_btn.setMinimumHeight(40)
        self.load_model_btn.setStyleSheet(self.get_button_style("primary"))
        self.load_model_btn.clicked.connect(self.load_selected_model)
        model_layout.addWidget(self.load_model_btn)

        model_group.setLayout(model_layout)
        left_panel.addWidget(model_group)

        # 2. Загрузка изображения
        image_group = QGroupBox("Изображение")
        image_layout = QVBoxLayout()

        self.image_preview = QLabel()
        self.image_preview.setMinimumSize(300, 300)
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setStyleSheet("""
            QLabel {
                border: 2px solid #ddd;
                border-radius: 8px;
                background-color: #f8f9fa;
            }
        """)
        self.image_preview.setText("Изображение не загружено")
        image_layout.addWidget(self.image_preview)

        btn_layout = QHBoxLayout()
        self.load_image_btn = QPushButton("📁 Загрузить изображение")
        self.load_image_btn.setMinimumHeight(40)
        self.load_image_btn.clicked.connect(self.load_image)
        self.load_image_btn.setStyleSheet(self.get_button_style("secondary"))

        self.clear_image_btn = QPushButton("Очистить")
        self.clear_image_btn.setMinimumHeight(40)
        self.clear_image_btn.clicked.connect(self.clear_image)
        self.clear_image_btn.setStyleSheet(self.get_button_style("danger"))

        btn_layout.addWidget(self.load_image_btn)
        btn_layout.addWidget(self.clear_image_btn)
        image_layout.addLayout(btn_layout)

        image_group.setLayout(image_layout)
        left_panel.addWidget(image_group)

        # 3. Прогресс
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_panel.addWidget(self.progress_bar)

        # 4. Информация
        info_group = QGroupBox("Информация")
        info_layout = QVBoxLayout()

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(150)
        self.info_text.setPlainText("Модель не загружена")
        info_layout.addWidget(self.info_text)

        info_group.setLayout(info_layout)
        left_panel.addWidget(info_group)

        # Правая панель - результаты
        right_panel = QVBoxLayout()

        # 1. Результаты предсказания
        results_group = QGroupBox("Результаты")
        results_layout = QVBoxLayout()

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(400)
        self.results_text.setPlainText("Загрузите модель и изображение для тестирования")
        results_layout.addWidget(self.results_text)

        results_group.setLayout(results_layout)
        right_panel.addWidget(results_group)

        # 2. Визуализация уверенности
        confidence_group = QGroupBox("Уверенность модели")
        confidence_layout = QVBoxLayout()

        self.confidence_widget = QWidget()
        self.confidence_widget.setMinimumHeight(200)
        confidence_layout.addWidget(self.confidence_widget)

        confidence_group.setLayout(confidence_layout)
        right_panel.addWidget(confidence_group)

        # 3. Кнопки действий
        action_group = QGroupBox("Действия")
        action_layout = QHBoxLayout()

        self.save_results_btn = QPushButton("💾 Сохранить результаты")
        self.save_results_btn.setMinimumHeight(40)
        self.save_results_btn.clicked.connect(self.save_results)
        self.save_results_btn.setEnabled(False)
        self.save_results_btn.setStyleSheet(self.get_button_style("success"))

        self.export_image_btn = QPushButton("📤 Экспорт с результатами")
        self.export_image_btn.setMinimumHeight(40)
        self.export_image_btn.clicked.connect(self.export_image_with_results)
        self.export_image_btn.setEnabled(False)
        self.export_image_btn.setStyleSheet(self.get_button_style("info"))

        self.batch_test_btn = QPushButton("🔄 Пакетное тестирование")
        self.batch_test_btn.setMinimumHeight(40)
        self.batch_test_btn.clicked.connect(self.batch_test)
        self.batch_test_btn.setStyleSheet(self.get_button_style("warning"))

        action_layout.addWidget(self.save_results_btn)
        action_layout.addWidget(self.export_image_btn)
        action_layout.addWidget(self.batch_test_btn)

        action_group.setLayout(action_layout)
        right_panel.addWidget(action_group)

        # Добавляем панели
        main_layout.addLayout(left_panel, 40)
        main_layout.addLayout(right_panel, 60)

        # Статус бар
        self.statusBar().showMessage("Готово к работе")

        # Устанавливаем стиль
        self.setStyleSheet(self.get_main_style())

    def get_button_style(self, button_type):
        """Возвращает стиль для кнопок"""
        styles = {
            "primary": """
                QPushButton {
                    background-color: #007bff;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #0056b3;
                }
                QPushButton:pressed {
                    background-color: #004085;
                }
            """,
            "secondary": """
                QPushButton {
                    background-color: #6c757d;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #545b62;
                }
                QPushButton:pressed {
                    background-color: #3d4348;
                }
            """,
            "success": """
                QPushButton {
                    background-color: #28a745;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #1e7e34;
                }
                QPushButton:pressed {
                    background-color: #155724;
                }
            """,
            "danger": """
                QPushButton {
                    background-color: #dc3545;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #bd2130;
                }
                QPushButton:pressed {
                    background-color: #721c24;
                }
            """,
            "warning": """
                QPushButton {
                    background-color: #ffc107;
                    color: #212529;
                    border: none;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #e0a800;
                }
                QPushButton:pressed {
                    background-color: #b38f00;
                }
            """,
            "info": """
                QPushButton {
                    background-color: #17a2b8;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #138496;
                }
                QPushButton:pressed {
                    background-color: #0c5460;
                }
            """
        }
        return styles.get(button_type, styles["primary"])

    def get_main_style(self):
        """Возвращает основной стиль приложения"""
        return """
            QMainWindow {
                background-color: #f5f7fa;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px 0 10px;
                color: #495057;
            }
            QTextEdit {
                border: 1px solid #ced4da;
                border-radius: 6px;
                padding: 8px;
                font-size: 13px;
                background-color: white;
            }
            QComboBox {
                border: 1px solid #ced4da;
                border-radius: 6px;
                padding: 8px;
                font-size: 13px;
                background-color: white;
            }
            QComboBox:focus {
                border-color: #80bdff;
                outline: 0;
                box-shadow: 0 0 0 0.2rem rgba(0,123,255,.25);
            }
            QProgressBar {
                border: 1px solid #ced4da;
                border-radius: 6px;
                text-align: center;
                background-color: white;
            }
            QProgressBar::chunk {
                background-color: #007bff;
                border-radius: 5px;
            }
        """

    def load_available_models(self):
        """Загружает список доступных моделей"""
        self.model_combo.clear()

        # Ищем модели в папке checkpoints
        checkpoints_dir = "checkpoints"
        if not os.path.exists(checkpoints_dir):
            self.statusBar().showMessage(f"Папка {checkpoints_dir} не найдена")
            return

        # Собираем все .pth файлы
        model_files = []
        for root, dirs, files in os.walk(checkpoints_dir):
            for file in files:
                if file.endswith('.pth'):
                    full_path = os.path.join(root, file)
                    model_files.append(full_path)

        if not model_files:
            self.model_combo.addItem("Модели не найдены")
            self.load_model_btn.setEnabled(False)
            return

        # Сортируем по дате изменения (новые сверху)
        model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        for model_path in model_files:
            filename = os.path.basename(model_path)
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            display_text = f"{filename} ({size_mb:.1f} MB)"
            self.model_combo.addItem(display_text, model_path)

        self.statusBar().showMessage(f"Найдено {len(model_files)} моделей")

    def load_selected_model(self):
        """Загружает выбранную модель"""
        if self.model_combo.currentIndex() < 0:
            return

        model_path = self.model_combo.currentData()
        if not model_path or not os.path.exists(model_path):
            QMessageBox.warning(self, "Ошибка", "Файл модели не найден")
            return

        # Показываем прогресс
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Индикатор без определенного диапазона

        # Загружаем в отдельном потоке
        QApplication.processEvents()  # Обновляем UI

        try:
            self.model, self.class_names, self.model_info = ModelLoader.load_model(model_path)

            # Обновляем информацию
            info_text = f"""Модель: {self.model_info['name']}
Архитектура: {self.model_info['architecture']}
Классов: {self.model_info['num_classes']}
Точность: {self.model_info['accuracy']}%
Эпоха обучения: {self.model_info['epoch']}
Дата: {self.model_info['timestamp']}

Имена классов: {len(self.class_names)}"""

            self.info_text.setPlainText(info_text)
            self.statusBar().showMessage(f"Модель загружена: {self.model_info['name']}")

            # Активируем кнопки
            self.load_image_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить модель:\n{str(e)}")
            self.statusBar().showMessage("Ошибка загрузки модели")

        finally:
            self.progress_bar.setVisible(False)

    def load_image(self):
        """Загружает изображение"""
        if self.model is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите модель!")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "",
            "Изображения (*.png *.jpg *.jpeg *.bmp *.gif);;Все файлы (*.*)"
        )

        if file_path:
            self.current_image_path = file_path
            self.process_image(file_path)

    def process_image(self, image_path):
        """Обрабатывает изображение и делает предсказание"""
        if not self.model:
            return

        # Показываем прогресс
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        QApplication.processEvents()

        try:
            # Отображаем превью
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_preview.setPixmap(scaled_pixmap)

            # Делаем предсказание
            result = ImageProcessor.predict_image(self.model, image_path, self.class_names)

            if result['success']:
                self.display_results(result, image_path)
                # Активируем кнопки
                self.save_results_btn.setEnabled(True)
                self.export_image_btn.setEnabled(True)
            else:
                QMessageBox.warning(self, "Ошибка", f"Не удалось обработать изображение:\n{result['error']}")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка обработки:\n{str(e)}")
            self.statusBar().showMessage("Ошибка обработки изображения")

        finally:
            self.progress_bar.setVisible(False)

    def display_results(self, result, image_path):
        """Отображает результаты предсказания"""
        predictions = result['predictions']

        # Формируем текст результатов
        text = "=" * 60 + "\n"
        text += "🚀 РЕЗУЛЬТАТЫ ПРЕДСКАЗАНИЯ\n"
        text += "=" * 60 + "\n\n"

        text += f"📁 Файл: {os.path.basename(image_path)}\n"
        text += f"📐 Размер: {result['image_size'][0]}x{result['image_size'][1]}\n"
        text += f"🕐 Время: {datetime.now().strftime('%H:%M:%S')}\n\n"

        text += "🏆 ТОП-5 РЕЗУЛЬТАТОВ:\n"
        text += "-" * 40 + "\n\n"

        for i, pred in enumerate(predictions, 1):
            confidence = pred['confidence']
            # Создаем визуальный индикатор
            bar_length = 30
            filled = int(bar_length * confidence / 100)
            bar = '█' * filled + '░' * (bar_length - filled)

            text += f"{i}. {pred['class']}\n"
            text += f"   Уверенность: {confidence:5.1f}% [{bar}]\n\n"

        best = predictions[0]
        text += "=" * 60 + "\n"
        text += f"✅ РЕКОМЕНДАЦИЯ: {best['class']} ({best['confidence']:.1f}%)\n"
        text += "=" * 60 + "\n\n"

        # Оценка уверенности
        if best['confidence'] > 80:
            text += "💪 Модель ОЧЕНЬ уверена в результате\n"
        elif best['confidence'] > 60:
            text += "👍 Модель уверена в результате\n"
        elif best['confidence'] > 40:
            text += "🤔 Модель умеренно уверена\n"
        else:
            text += "❓ Модель не уверена, рассмотрите другие варианты\n"

        # Сравнение со вторым результатом
        if len(predictions) > 1:
            diff = best['confidence'] - predictions[1]['confidence']
            if diff < 10:
                text += f"⚖️  Разница со вторым местом всего {diff:.1f}% - результат неоднозначный\n"

        self.results_text.setPlainText(text)
        self.statusBar().showMessage(f"Результат: {best['class']} ({best['confidence']:.1f}%)")

        # Визуализируем уверенность
        self.plot_confidence(predictions)

    def plot_confidence(self, predictions):
        """Создает график уверенности"""
        try:
            # Создаем изображение для отображения
            width = 500
            height = 200
            bar_height = 30
            margin = 20

            # Создаем QPixmap для отрисовки
            pixmap = QPixmap(width, height)
            pixmap.fill(Qt.white)

            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)

            # Устанавливаем шрифт
            font = QFont("Arial", 10)
            painter.setFont(font)

            # Рисуем заголовок
            painter.setPen(QColor(0, 0, 0))
            painter.drawText(10, 20, "Уверенность модели:")

            # Рисуем столбцы
            for i, pred in enumerate(predictions[:5]):  # Первые 5
                y = 40 + i * (bar_height + 10)
                confidence = pred['confidence']

                # Текст класса
                class_text = f"{i + 1}. {pred['class'][:20]}{'...' if len(pred['class']) > 20 else ''}"
                painter.drawText(10, y + 20, class_text)

                # Полоса уверенности
                bar_width = int((width - 200) * confidence / 100)

                # Цвет в зависимости от уверенности
                if confidence > 80:
                    color = QColor(40, 167, 69)  # Зеленый
                elif confidence > 60:
                    color = QColor(23, 162, 184)  # Голубой
                elif confidence > 40:
                    color = QColor(255, 193, 7)  # Желтый
                else:
                    color = QColor(220, 53, 69)  # Красный

                # Рисуем фон полосы
                painter.setPen(QColor(200, 200, 200))
                painter.setBrush(QColor(240, 240, 240))
                painter.drawRect(200, y, width - 220, bar_height)

                # Рисуем заполненную часть
                painter.setPen(Qt.NoPen)
                painter.setBrush(color)
                painter.drawRect(200, y, bar_width, bar_height)

                # Текст с процентом
                painter.setPen(QColor(0, 0, 0))
                percent_text = f"{confidence:.1f}%"
                text_width = painter.fontMetrics().width(percent_text)
                painter.drawText(width - text_width - 10, y + 20, percent_text)

            painter.end()

            # Отображаем в виджете
            self.confidence_widget = QLabel()
            self.confidence_widget.setPixmap(pixmap)

            # Заменяем старый виджет
            old_widget = self.confidence_widget
            layout = self.confidence_widget.parent().layout()
            layout.replaceWidget(old_widget, self.confidence_widget)
            old_widget.deleteLater()

        except Exception as e:
            print(f"Ошибка отрисовки графика: {e}")

    def clear_image(self):
        """Очищает изображение и результаты"""
        self.image_preview.clear()
        self.image_preview.setText("Изображение не загружено")
        self.results_text.clear()
        self.results_text.setPlainText("Загрузите модель и изображение для тестирования")
        self.current_image_path = None

        # Деактивируем кнопки
        self.save_results_btn.setEnabled(False)
        self.export_image_btn.setEnabled(False)

        # Очищаем график
        self.confidence_widget = QWidget()
        self.confidence_widget.setMinimumHeight(200)
        layout = self.confidence_widget.parent().layout()
        old_widget = layout.itemAt(0).widget()
        layout.replaceWidget(old_widget, self.confidence_widget)
        old_widget.deleteLater()

        self.statusBar().showMessage("Готово")

    def save_results(self):
        """Сохраняет результаты в файл"""
        if not self.current_image_path:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результаты", "results.txt",
            "Текстовые файлы (*.txt);;Все файлы (*.*)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.results_text.toPlainText())

                QMessageBox.information(self, "Успех", f"Результаты сохранены в:\n{file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл:\n{str(e)}")

    def export_image_with_results(self):
        """Экспортирует изображение с наложенными результатами"""
        if not self.current_image_path or not self.model:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить изображение с результатами", "result_image.jpg",
            "Изображения (*.jpg *.png);;Все файлы (*.*)"
        )

        if file_path:
            try:
                # Открываем изображение
                image = Image.open(self.current_image_path).convert('RGB')

                # Создаем копию для рисования
                draw_image = image.copy()
                draw = ImageDraw.Draw(draw_image)

                # Загружаем шрифт
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()

                # Добавляем текст с результатами
                results_text = self.results_text.toPlainText()
                lines = results_text.split('\n')

                # Выбираем только ключевую информацию
                key_lines = []
                for line in lines:
                    if any(keyword in line for keyword in ["РЕКОМЕНДАЦИЯ", "Уверенность:", "1.", "2.", "3."]):
                        key_lines.append(line)

                # Рисуем фон для текста
                text = "\n".join(key_lines[:8])  # Первые 8 строк
                bbox = draw.textbbox((10, 10), text, font=font)
                draw.rectangle([bbox[0] - 5, bbox[1] - 5, bbox[2] + 5, bbox[3] + 5],
                               fill=(255, 255, 255, 200))

                # Рисуем текст
                draw.text((10, 10), text, fill=(0, 0, 0), font=font)

                # Сохраняем
                draw_image.save(file_path)

                QMessageBox.information(self, "Успех", f"Изображение сохранено в:\n{file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить изображение:\n{str(e)}")

    def batch_test(self):
        """Пакетное тестирование изображений"""
        if not self.model:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите модель!")
            return

        folder_path = QFileDialog.getExistingDirectory(
            self, "Выберите папку с изображениями", ""
        )

        if not folder_path:
            return

        # Получаем список изображений
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        image_files = []
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(folder_path, file))

        if not image_files:
            QMessageBox.warning(self, "Ошибка", "В папке нет изображений")
            return

        # Диалог прогресса
        progress_dialog = QProgressDialog("Обработка изображений...", "Отмена", 0, len(image_files), self)
        progress_dialog.setWindowTitle("Пакетная обработка")
        progress_dialog.setWindowModality(Qt.WindowModal)

        results = []
        for i, image_path in enumerate(image_files):
            progress_dialog.setValue(i)
            progress_dialog.setLabelText(f"Обработка: {os.path.basename(image_path)}")

            if progress_dialog.wasCanceled():
                break

            QApplication.processEvents()

            try:
                result = ImageProcessor.predict_image(self.model, image_path, self.class_names)
                if result['success']:
                    results.append({
                        'file': os.path.basename(image_path),
                        'prediction': result['best_prediction']['class'],
                        'confidence': result['best_prediction']['confidence']
                    })
            except:
                pass

        progress_dialog.close()

        # Показываем результаты
        if results:
            result_text = "📊 РЕЗУЛЬТАТЫ ПАКЕТНОЙ ОБРАБОДКИ\n"
            result_text += "=" * 40 + "\n\n"
            result_text += f"Обработано изображений: {len(results)}\n\n"

            for res in results:
                result_text += f"• {res['file']}: {res['prediction']} ({res['confidence']:.1f}%)\n"

            self.results_text.setPlainText(result_text)
            self.statusBar().showMessage(f"Пакетная обработка завершена: {len(results)} изображений")
        else:
            self.statusBar().showMessage("Пакетная обработка не дала результатов")

    def closeEvent(self, event):
        """Обработка закрытия приложения"""
        reply = QMessageBox.question(
            self, 'Подтверждение',
            'Вы уверены, что хотите выйти?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


def main():
    """Запуск приложения"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Устанавливаем иконку приложения
    app.setWindowIcon(QIcon('icon.png') if os.path.exists('icon.png') else QIcon())

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()