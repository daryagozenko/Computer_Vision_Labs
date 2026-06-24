from PIL import Image, ImageEnhance, ExifTags, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import gc
import os

class ImageProcessor:
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.histogram_figure = None
        self.current_brightness = 1.0
        self.current_contrast = 1.0
        self.current_saturation = 1.0
        self.original_max_vals = {'r': 1.0, 'g': 1.0, 'b': 1.0, 'gray': 1.0}
        self.is_grayscale_mode = False
        self.rotation_count = 0

    def load_image(self, file_path):
        try:
            self.close_images()
            self.original_image = Image.open(file_path)
            self.original_image = self._apply_exif_orientation(self.original_image)
            self.processed_image = self.original_image.copy()
            self.current_brightness = 1.0
            self.current_contrast = 1.0
            self.current_saturation = 1.0
            self.is_grayscale_mode = False
            self.rotation_count = 0

            self._calculate_original_max_vals()
            return True
        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            return False

    def _apply_exif_orientation(self, image):
        try:
            exif = image._getexif()
            if exif:
                orientation = exif.get(0x0112)
                if orientation:
                    if orientation == 2:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation == 3:
                        image = image.rotate(180)
                    elif orientation == 4:
                        image = image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation == 5:
                        image = image.rotate(-90).transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation == 6:
                        image = image.rotate(-90)
                    elif orientation == 7:
                        image = image.rotate(90).transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation == 8:
                        image = image.rotate(90)
        except:
            pass
        return image

    def _calculate_original_max_vals(self):
        if self.original_image.mode == 'L':
            hist = self.original_image.histogram()[:256]
            self.original_max_vals['gray'] = max(hist) if max(hist) > 0 else 1.0
        else:
            r, g, b = self.original_image.split()
            r_hist = r.histogram()[:256]
            g_hist = g.histogram()[:256]
            b_hist = b.histogram()[:256]

            self.original_max_vals['r'] = max(r_hist) if max(r_hist) > 0 else 1.0
            self.original_max_vals['g'] = max(g_hist) if max(g_hist) > 0 else 1.0
            self.original_max_vals['b'] = max(b_hist) if max(b_hist) > 0 else 1.0

            r.close()
            g.close()
            b.close()

    def __del__(self):
        """Деструктор для очистки ресурсов"""
        self.close_images()

    def close_images(self):
        """Улучшенная очистка ресурсов"""
        resources = [
            self.original_image,
            self.processed_image,
            self.histogram_figure
        ]

        for resource in resources:
            if resource:
                try:
                    if hasattr(resource, 'close'):
                        resource.close()
                    elif hasattr(resource, 'figure'):  # Для matplotlib figures
                        plt.close(resource)
                except:
                    pass

        self.original_image = None
        self.processed_image = None
        self.histogram_figure = None

        # Очищаем numpy кэш
        try:
            np.zeros(0)  # Освобождает внутренний кэш numpy
        except:
            pass

        gc.collect()

    def get_image_info(self):
        if not self.original_image:
            return None

        try:
            file_size = os.path.getsize(self.original_image.filename) if self.original_image.filename else "Неизвестно"
        except:
            file_size = "Неизвестно"

        exif_data = self._get_exif_data()

        return {
            'Размер на диске': f"{file_size} байт",
            'Разрешение': f"{self.original_image.width} x {self.original_image.height}",
            'Глубина цвета': f"{self.original_image.bits} бит" if hasattr(self.original_image, 'bits') else "Неизвестно",
            'Формат файла': self.original_image.format,
            'Цветовая модель': self.original_image.mode,
            'EXIF данные': exif_data
        }

    def _get_exif_data(self):
        exif_data = {}
        try:
            if not hasattr(self.original_image, 'getexif'):
                return {"Информация": "Изображение не содержит EXIF данных"}

            exif = self.original_image.getexif()

            if not exif:
                return {"Информация": "EXIF данные отсутствуют"}
            if exif:
                for tag, value in exif.items():
                    tag_name = ExifTags.TAGS.get(tag, tag)
                    if tag_name in ['DateTime', 'Make', 'Model', 'ExposureTime', 'FNumber',
                                    'ISOSpeedRatings', 'FocalLength', 'Software']:
                        if isinstance(value, bytes):
                            try:
                                value = value.decode('utf-8', errors='ignore')
                            except:
                                value = str(value)
                        exif_data[tag_name] = value
                    if len(exif_data) >= 5:
                        break
        except Exception as e:
            print(f"Ошибка чтения EXIF: {e}")
            exif_data = {"Ошибка": "Не удалось прочитать EXIF"}
        return exif_data

    def convert_grayscale(self):
        if self.processed_image:
            self.processed_image = self.processed_image.convert('L')
            self.current_saturation = 0.0
            self.is_grayscale_mode = True

    def convert_to_rgb(self):
        if self.processed_image and self.original_image:
            self.processed_image = self.original_image.copy()
            self.is_grayscale_mode = False
            self._apply_current_adjustments()

    def is_grayscale(self):
        return self.is_grayscale_mode

    def adjust_brightness(self, factor):
        self.current_brightness = factor / 50.0
        self._apply_current_adjustments()

    def adjust_contrast(self, factor):
        self.current_contrast = factor / 50.0
        self._apply_current_adjustments()

    def adjust_saturation(self, factor):
        self.current_saturation = factor / 50.0
        self._apply_current_adjustments()

    def _apply_current_adjustments(self):
        if not self.original_image:
            return

        try:
            # Освобождаем предыдущее обработанное изображение
            if self.processed_image and self.processed_image != self.original_image:
                self.processed_image.close()

            if self.is_grayscale_mode:
                # Создаем grayscale версию оригинального изображения
                base_image = self.original_image.copy().convert('L')

                # Применяем корректировки
                if self.current_contrast != 1.0:
                    enhancer = ImageEnhance.Contrast(base_image)
                    base_image = enhancer.enhance(self.current_contrast)
                    enhancer = None  # Освобождаем enhancer

                if self.current_brightness != 1.0:
                    enhancer = ImageEnhance.Brightness(base_image)
                    base_image = enhancer.enhance(self.current_brightness)
                    enhancer = None

                self.processed_image = base_image
                return

            # Для цветного режима
            base_image = self.original_image.copy()

            # Применяем контраст
            if self.current_contrast != 1.0:
                temp_image = base_image
                base_image = self._apply_contrast_soft_optimized(base_image, self.current_contrast)
                if temp_image != self.original_image:
                    temp_image.close()

            # Применяем насыщенность
            if self.current_saturation != 1.0:
                temp_image = base_image
                base_image = self._apply_saturation_optimized(base_image, self.current_saturation)
                if temp_image != self.original_image:
                    temp_image.close()

            # Применяем яркость
            if self.current_brightness != 1.0:
                temp_image = base_image
                enhancer = ImageEnhance.Brightness(base_image)
                base_image = enhancer.enhance(self.current_brightness)
                enhancer = None
                if temp_image != self.original_image:
                    temp_image.close()

            self.processed_image = base_image

        except Exception as e:
            print(f"Ошибка применения корректировок: {e}")
            if self.original_image:
                self.processed_image = self.original_image.copy()


    def _apply_saturation_optimized(self, image, factor):
        """Оптимизированное применение насыщенности"""
        try:
            if image.mode != 'RGB':
                img_rgb = image.convert('RGB')
            else:
                img_rgb = image

            if factor == 0.0:
                result = img_rgb.convert('L').convert('RGB')
                if img_rgb != image:
                    img_rgb.close()
                return result

            hsv = img_rgb.convert('HSV')
            h, s, v = hsv.split()

            # Используем более эффективную обработку
            s_array = np.array(s, dtype=np.float32)
            s_array *= factor
            np.clip(s_array, 0, 255, out=s_array)

            s_enhanced = Image.fromarray(s_array.astype(np.uint8), 'L')
            hsv_enhanced = Image.merge('HSV', (h, s_enhanced, v))
            result = hsv_enhanced.convert('RGB')

            # Освобождаем временные объекты
            h.close();
            s.close();
            v.close()
            s_enhanced.close()
            hsv_enhanced.close()
            if img_rgb != image:
                img_rgb.close()
            hsv.close()

            return result

        except Exception as e:
            print(f"Ошибка применения насыщенности: {e}")
            enhancer = ImageEnhance.Color(image)
            result = enhancer.enhance(factor)
            enhancer = None
            return result

    def _apply_contrast_soft_optimized(self, image, factor):
        """Оптимизированное применение контраста"""
        try:
            if image.mode == 'L':
                img_array = np.array(image, dtype=np.float32)
                mean = np.mean(img_array)
                adjusted_factor = 1.0 + (factor - 1.0) * 0.7
                img_array = mean + (img_array - mean) * adjusted_factor
                np.clip(img_array, 0, 255, out=img_array)
                return Image.fromarray(img_array.astype(np.uint8), 'L')
            else:
                enhancer = ImageEnhance.Contrast(image)
                soft_factor = max(0.3, min(2.0, factor))
                result = enhancer.enhance(soft_factor)
                enhancer = None
                return result

        except Exception as e:
            print(f"Ошибка мягкой регулировки контраста: {e}")
            enhancer = ImageEnhance.Contrast(image)
            result = enhancer.enhance(max(0.5, min(1.5, factor)))
            enhancer = None
            return result

    def apply_all_adjustments(self, brightness_factor, contrast_factor, saturation_factor):
        self.current_brightness = brightness_factor
        self.current_contrast = contrast_factor
        self.current_saturation = saturation_factor
        self._apply_current_adjustments()

    def rotate_90(self):
        if self.processed_image:
            self.processed_image = self.processed_image.transpose(Image.ROTATE_90)
            self.rotation_count = (self.rotation_count + 1) % 4

    def apply_linear_correction(self, a=1.0, b=0.0):
        if self.processed_image and self.is_grayscale_mode:
            img_array = np.array(self.processed_image, dtype=np.float32)
            img_array = a * img_array + b
            img_array = np.clip(img_array, 0, 255)
            self.processed_image = Image.fromarray(img_array.astype(np.uint8), 'L')

    def apply_nonlinear_correction(self, gamma=1.0):
        if self.processed_image and self.is_grayscale_mode:
            img_array = np.array(self.processed_image, dtype=np.float32)
            img_array = 255 * np.power(img_array / 255.0, gamma)
            img_array = np.clip(img_array, 0, 255)
            self.processed_image = Image.fromarray(img_array.astype(np.uint8), 'L')

    def reset_all_adjustments(self):
        if self.original_image:
            self.processed_image = self.original_image.copy()
            self.current_brightness = 1.0
            self.current_contrast = 1.0
            self.current_saturation = 1.0
            self.is_grayscale_mode = False
            self.rotation_count = 0

    def create_histogram(self, channel='all'):
        if not self.processed_image:
            return None

        # Закрываем предыдущую фигуру
        if self.histogram_figure:
            plt.close(self.histogram_figure)
            self.histogram_figure = None

        try:
            # Создаем новую фигуру
            self.histogram_figure, ax = plt.subplots(figsize=(8, 4))
            self.histogram_figure.patch.set_facecolor('#1E1E1E')
            ax.set_facecolor('#1E1E1E')
            ax.tick_params(colors='white', labelsize=9)

            for spine in ax.spines.values():
                spine.set_color('#444444')
                spine.set_linewidth(1)

            show_grayscale = (self.processed_image.mode == 'L' or self.is_grayscale_mode or
                              (self.current_saturation == 0.0 and not self.is_grayscale_mode))

            if show_grayscale:
                # ВОЗВРАЩАЕМ СТАРУЮ ЛОГИКУ ДЛЯ GRAYSCALE
                if self.processed_image.mode == 'L':
                    gray_hist = self.processed_image.histogram()[:256]
                    r_hist = gray_hist.copy()
                    g_hist = gray_hist.copy()
                    b_hist = gray_hist.copy()
                else:
                    gray_img = self.processed_image.convert('L')
                    gray_hist = gray_img.histogram()[:256]
                    gray_img.close()
                    r_hist = gray_hist.copy()
                    g_hist = gray_hist.copy()
                    b_hist = gray_hist.copy()

                r_norm = np.array(r_hist, dtype=float) / self.original_max_vals['gray'] if self.original_max_vals[
                                                                                               'gray'] > 0 else r_hist
                g_norm = np.array(g_hist, dtype=float) / self.original_max_vals['gray'] if self.original_max_vals[
                                                                                               'gray'] > 0 else g_hist
                b_norm = np.array(b_hist, dtype=float) / self.original_max_vals['gray'] if self.original_max_vals[
                                                                                               'gray'] > 0 else b_hist

                x = np.arange(256)
                light_gray = '#CCCCCC'
                medium_gray = '#999999'
                dark_gray = '#666666'

                if channel in ['all', 'r']:
                    ax.fill_between(x, r_norm, color=light_gray, alpha=0.4, label='Gray (R)')
                    ax.plot(x, r_norm, color=light_gray, linewidth=1.5, alpha=0.9)

                if channel in ['all', 'g']:
                    ax.fill_between(x, g_norm, color=medium_gray, alpha=0.4, label='Gray (G)')
                    ax.plot(x, g_norm, color=medium_gray, linewidth=1.5, alpha=0.9)

                if channel in ['all', 'b']:
                    ax.fill_between(x, b_norm, color=dark_gray, alpha=0.4, label='Gray (B)')
                    ax.plot(x, b_norm, color=dark_gray, linewidth=1.5, alpha=0.9)

                if channel == 'all':
                    total_norm = (r_norm + g_norm + b_norm) / 3
                    ax.fill_between(x, total_norm, color='#FFFFFF', alpha=0.15)

                ax.legend(facecolor='#2D2D2D', edgecolor='#555555', labelcolor='white',
                          fontsize=9, loc='upper right', framealpha=0.9)
                ax.set_title('Grayscale Гистограмма', color='white', pad=15, fontsize=12, fontweight='bold')

            else:
                # ОПТИМИЗИРОВАННАЯ ОБРАБОТКА RGB (оставляем новую)
                if self.processed_image.mode != 'RGB':
                    with self.processed_image.convert('RGB') as img_rgb:
                        r, g, b = img_rgb.split()
                else:
                    r, g, b = self.processed_image.split()

                # Вычисляем гистограммы для каждого канала
                try:
                    r_array = np.array(r)
                    g_array = np.array(g)
                    b_array = np.array(b)

                    r_hist, _ = np.histogram(r_array, bins=256, range=(0, 255))
                    g_hist, _ = np.histogram(g_array, bins=256, range=(0, 255))
                    b_hist, _ = np.histogram(b_array, bins=256, range=(0, 255))

                    # Освобождаем массивы сразу после использования
                    del r_array, g_array, b_array

                finally:
                    # Всегда закрываем временные изображения
                    r.close()
                    g.close()
                    b.close()

                # Нормализуем гистограммы
                r_norm = r_hist.astype(float) / (
                    self.original_max_vals['r'] if self.original_max_vals['r'] > 0 else 1.0)
                g_norm = g_hist.astype(float) / (
                    self.original_max_vals['g'] if self.original_max_vals['g'] > 0 else 1.0)
                b_norm = b_hist.astype(float) / (
                    self.original_max_vals['b'] if self.original_max_vals['b'] > 0 else 1.0)

                x = np.arange(256)

                # Рисуем только выбранные каналы
                if channel in ['all', 'r']:
                    ax.fill_between(x, r_norm, color='#FF6B6B', alpha=0.4, label='Red')
                    ax.plot(x, r_norm, color='#FF6B6B', linewidth=1.5, alpha=0.9)

                if channel in ['all', 'g']:
                    ax.fill_between(x, g_norm, color='#4ECDC4', alpha=0.4, label='Green')
                    ax.plot(x, g_norm, color='#4ECDC4', linewidth=1.5, alpha=0.9)

                if channel in ['all', 'b']:
                    ax.fill_between(x, b_norm, color='#45B7D1', alpha=0.4, label='Blue')
                    ax.plot(x, b_norm, color='#45B7D1', linewidth=1.5, alpha=0.9)

                if channel == 'all':
                    total_norm = (r_norm + g_norm + b_norm) / 3
                    ax.fill_between(x, total_norm, color='#FFFFFF', alpha=0.15)

                ax.legend(facecolor='#2D2D2D', edgecolor='#555555', labelcolor='white',
                          fontsize=9, loc='upper right', framealpha=0.9)
                ax.set_title('RGB Гистограмма', color='white', pad=15, fontsize=12, fontweight='bold')

            # Настройки осей
            ax.set_xlim(0, 255)
            ax.set_ylim(0, 1.2)
            ax.grid(True, alpha=0.15, color='white', linestyle='--', linewidth=0.5)
            ax.set_xlabel('Уровень яркости', color='white', labelpad=10, fontsize=10)
            ax.set_ylabel('Относительная частота', color='white', labelpad=10, fontsize=10)
            ax.set_xticks([0, 64, 128, 192, 255])
            ax.set_xticklabels(['0', '64', '128', '192', '255'])
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

            plt.tight_layout()

            # Принудительная сборка мусора
            gc.collect()

            return self.histogram_figure

        except Exception as e:
            print(f"Ошибка создания гистограммы: {e}")
            if self.histogram_figure:
                plt.close(self.histogram_figure)
                self.histogram_figure = None
            return None

    def save_image(self, file_path):
        if self.processed_image:
            try:
                self.processed_image.save(file_path)
                return True  # Успешное сохранение
            except Exception as e:
                print(f"Ошибка при сохранении: {e}")
                return False  # Ошибка сохранения
        return False  # Нет изображения для сохранения

    def get_image_stats(self):
        if not self.processed_image:
            return "Нет изображения"

        stats = f"Режим: {self.processed_image.mode}\n"
        stats += f"Размер: {self.processed_image.width} x {self.processed_image.height}\n"
        stats += f"Яркость: {self.current_brightness:.2f}\n"
        stats += f"Контраст: {self.current_contrast:.2f}\n"
        stats += f"Насыщенность: {self.current_saturation:.2f}\n"
        stats += f"Grayscale режим: {'Да' if self.is_grayscale_mode else 'Нет'}\n"

        if self.processed_image.mode == 'L':
            pixels = np.array(self.processed_image).flatten()
            stats += f"Диапазон: {np.min(pixels)} - {np.max(pixels)}\n"
            stats += f"Среднее: {np.mean(pixels):.1f}\n"
            stats += f"Стандартное отклонение: {np.std(pixels):.1f}\n"
        else:
            r, g, b = self.processed_image.split()
            r_arr, g_arr, b_arr = np.array(r), np.array(g), np.array(b)

            stats += f"Red: {np.min(r_arr)}-{np.max(r_arr)} (mean: {np.mean(r_arr):.1f})\n"
            stats += f"Green: {np.min(g_arr)}-{np.max(g_arr)} (mean: {np.mean(g_arr):.1f})\n"
            stats += f"Blue: {np.min(b_arr)}-{np.max(b_arr)} (mean: {np.mean(b_arr):.1f})\n"

            r.close(); g.close(); b.close()

        return stats

    def get_current_values(self):
        return {
            'brightness': int(self.current_brightness * 50),
            'contrast': int(self.current_contrast * 50),
            'saturation': int(self.current_saturation * 50)
        }