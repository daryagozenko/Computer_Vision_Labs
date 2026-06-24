# utils.py
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from PIL import Image
import json


class EarlyStopping:
    """Ранняя остановка для предотвращения переобучения"""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): Сколько эпох ждать ухудшения
            verbose (bool): Выводить ли сообщения
            delta (float): Минимальное изменение для улучшения
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Сохраняет модель при улучшении"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), 'checkpoints/early_stopping_checkpoint.pth')
        self.val_loss_min = val_loss


def plot_training_history(history, save_path=None):
    """
    Визуализирует историю обучения
    Args:
        history (dict): История обучения
        save_path (str): Путь для сохранения графика
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
    axes[0, 1].plot(history['val_acc'], label='Val Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Learning Rate
    axes[1, 0].plot(history['learning_rate'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True)

    # Пустая область для дополнительной информации
    axes[1, 1].axis('off')
    if history['train_acc']:
        best_train_acc = max(history['train_acc'])
        best_val_acc = max(history['val_acc'])
        final_train_acc = history['train_acc'][-1]
        final_val_acc = history['val_acc'][-1]

        info_text = f"Best Train Acc: {best_train_acc:.2f}%\n"
        info_text += f"Best Val Acc: {best_val_acc:.2f}%\n"
        info_text += f"Final Train Acc: {final_train_acc:.2f}%\n"
        info_text += f"Final Val Acc: {final_val_acc:.2f}%\n"
        info_text += f"Total Epochs: {len(history['train_acc'])}"

        axes[1, 1].text(0.1, 0.5, info_text, fontsize=12,
                        verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"График сохранен: {save_path}")

    plt.show()


def predict_image(model, image_path, class_names, transform=None, device='cpu'):
    """
    Предсказывает класс для одного изображения
    Args:
        model: Обученная модель
        image_path (str): Путь к изображению
        class_names (list): Список названий классов
        transform: Преобразования для изображения
        device: Устройство для вычислений
    Returns:
        dict: Результаты предсказания
    """
    # Загружаем изображение
    image = Image.open(image_path).convert('RGB')

    # Применяем преобразования
    if transform is None:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Предсказание
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_class = torch.topk(probabilities, len(class_names))

    # Формируем результат
    results = {
        'top_prediction': {
            'class': class_names[top_class[0].item()],
            'confidence': top_prob[0].item() * 100
        },
        'all_predictions': []
    }

    for i in range(len(top_prob)):
        results['all_predictions'].append({
            'class': class_names[top_class[i].item()],
            'confidence': top_prob[i].item() * 100,
            'rank': i + 1
        })

    return results


def visualize_predictions(image_path, predictions, save_path=None):
    """
    Визуализирует предсказания на изображении
    Args:
        image_path (str): Путь к изображению
        predictions (dict): Результаты предсказания
        save_path (str): Путь для сохранения
    """
    # Загружаем изображение
    image = Image.open(image_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Изображение
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    # Предсказания
    classes = []
    confidences = []

    for pred in predictions['all_predictions'][:5]:  # Топ-5
        classes.append(pred['class'])
        confidences.append(pred['confidence'])

    y_pos = np.arange(len(classes))
    axes[1].barh(y_pos, confidences, align='center', alpha=0.8)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(classes)
    axes[1].set_xlabel('Confidence (%)')
    axes[1].set_title('Top-5 Predictions')
    axes[1].invert_yaxis()  # Самый уверенный сверху

    # Добавляем значения на столбцы
    for i, v in enumerate(confidences):
        axes[1].text(v + 1, i, f'{v:.1f}%', va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')

    plt.show()


def save_config(config, filepath='config.json'):
    """Сохраняет конфигурацию в JSON файл"""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Конфигурация сохранена: {filepath}")


def load_config(filepath='config.json'):
    """Загружает конфигурацию из JSON файла"""
    with open(filepath, 'r') as f:
        config = json.load(f)
    print(f"Конфигурация загружена: {filepath}")
    return config


def count_parameters(model):
    """Считает количество параметров модели"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }


if __name__ == "__main__":
    # Пример использования утилит
    print("Тестирование утилит...")

    # Пример конфигурации
    config = {
        'model_name': 'resnet50',
        'learning_rate': 0.001,
        'batch_size': 32
    }

    # Сохраняем и загружаем конфигурацию
    save_config(config, 'test_config.json')
    loaded_config = load_config('test_config.json')

    print(f"\nЗагруженная конфигурация: {loaded_config}")

    # Удаляем тестовый файл
    if os.path.exists('test_config.json'):
        os.remove('test_config.json')
        print("Тестовый файл конфигурации удален")