# augment_dataset.py
import os
import torch
from torchvision import transforms
from PIL import Image
import random
import argparse


class ImageAugmentor:
    """Класс для аугментации изображений"""

    def __init__(self, output_size=224):
        self.output_size = output_size

        # Определяем аугментации
        self.augmentations = [
            self.random_horizontal_flip,
            self.random_rotation,
            self.color_jitter,
            self.random_crop,
            self.random_translate,
            self.random_perspective,
            self.random_blur,
            self.random_noise,
        ]

    def random_horizontal_flip(self, image):
        if random.random() > 0.5:
            return transforms.functional.hflip(image)
        return image

    def random_rotation(self, image):
        angle = random.uniform(-15, 15)
        return transforms.functional.rotate(image, angle)

    def color_jitter(self, image):
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        saturation = random.uniform(0.8, 1.2)

        image = transforms.functional.adjust_brightness(image, brightness)
        image = transforms.functional.adjust_contrast(image, contrast)
        image = transforms.functional.adjust_saturation(image, saturation)
        return image

    def random_crop(self, image):
        scale = random.uniform(0.8, 1.0)
        new_size = int(self.output_size * scale)
        image = transforms.functional.resize(image, (new_size, new_size))

        # Случайный кроп до исходного размера
        return transforms.functional.center_crop(image, self.output_size)

    def random_translate(self, image):
        translate_x = random.uniform(-0.1, 0.1) * self.output_size
        translate_y = random.uniform(-0.1, 0.1) * self.output_size
        return transforms.functional.affine(
            image, angle=0, translate=(translate_x, translate_y),
            scale=1.0, shear=0
        )

    def random_perspective(self, image):
        if random.random() > 0.7:
            return transforms.functional.perspective(
                image,
                startpoints=[(0, 0), (self.output_size, 0),
                             (self.output_size, self.output_size), (0, self.output_size)],
                endpoints=[(random.randint(0, 20), random.randint(0, 20)),
                           (self.output_size - random.randint(0, 20), random.randint(0, 20)),
                           (self.output_size - random.randint(0, 20), self.output_size - random.randint(0, 20)),
                           (random.randint(0, 20), self.output_size - random.randint(0, 20))]
            )
        return image

    def random_blur(self, image):
        if random.random() > 0.8:
            return transforms.functional.gaussian_blur(image, kernel_size=3)
        return image

    def random_noise(self, image):
        if random.random() > 0.9:
            import numpy as np
            img_array = np.array(image).astype(np.float32)
            noise = np.random.normal(0, 10, img_array.shape).astype(np.float32)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(img_array)
        return image

    def augment_image(self, image, num_augmentations=3):
        """Применяет аугментации к изображению"""
        augmented_images = [image]

        for i in range(num_augmentations):
            aug_image = image.copy()

            # Применяем случайные аугментации
            num_ops = random.randint(2, 4)
            selected_ops = random.sample(self.augmentations, num_ops)

            for op in selected_ops:
                aug_image = op(aug_image)

            # Ресайз к нужному размеру
            aug_image = transforms.functional.resize(aug_image, (self.output_size, self.output_size))
            augmented_images.append(aug_image)

        return augmented_images


def augment_dataset(data_dir, augment_per_image=3, output_size=224):
    """
    Увеличивает датасет с помощью аугментации
    """
    print(f"🎨 Аугментация датасета в {data_dir}...")
    print(f"Увеличение в {augment_per_image + 1} раз")

    augmentor = ImageAugmentor(output_size)

    total_original = 0
    total_augmented = 0

    for split in ['train', 'val']:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            print(f"  Пропускаем {split}: папка не найдена")
            continue

        print(f"\n{split.upper()}:")

        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                continue

            # Получаем только оригинальные изображения (не аугментированные)
            original_images = [f for f in os.listdir(class_path)
                               if f.endswith(('.jpg', '.jpeg', '.png'))
                               and not f.startswith('aug_')]

            if not original_images:
                continue

            print(f"  {class_name}: {len(original_images)} исходных изображений")
            total_original += len(original_images)

            for img_name in original_images:
                img_path = os.path.join(class_path, img_name)

                try:
                    # Загружаем изображение
                    image = Image.open(img_path).convert('RGB')

                    # Аугментируем
                    augmented_images = augmentor.augment_image(image, augment_per_image)

                    # Сохраняем аугментированные версии
                    for i, aug_image in enumerate(augmented_images[1:], 1):  # Пропускаем оригинал
                        aug_name = f"aug_{i:02d}_{img_name}"
                        aug_path = os.path.join(class_path, aug_name)
                        aug_image.save(aug_path)
                        total_augmented += 1

                except Exception as e:
                    print(f"    Ошибка при аугментации {img_name}: {e}")

    print(f"\n✅ Аугментация завершена!")
    print(f"Исходных изображений: {total_original}")
    print(f"Аугментированных изображений: {total_augmented}")
    print(f"Всего изображений: {total_original + total_augmented}")
    print(f"Увеличение в {(total_original + total_augmented) / total_original:.1f} раз")


def main():
    parser = argparse.ArgumentParser(description='Аугментация датасета')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Папка с датасетом')
    parser.add_argument('--augment-factor', type=int, default=3,
                        help='Во сколько раз увеличить данные')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Размер изображений')

    args = parser.parse_args()

    augment_dataset(args.data_dir, args.augment_factor, args.image_size)


if __name__ == "__main__":
    main()