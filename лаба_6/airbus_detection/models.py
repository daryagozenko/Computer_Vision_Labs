# models.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights, MobileNet_V2_Weights


class SimpleCNN(nn.Module):
    """Простая CNN архитектура с нуля"""

    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()

        # Сверточные слои
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Полносвязные слои
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class AircraftClassifier(nn.Module):
    """Основной классификатор с поддержкой разных архитектур и правильным fine-tuning"""

    def __init__(self, num_classes=15, model_name='resnet50', pretrained=True, fine_tune_mode='last'):
        """
        Args:
            num_classes (int): Количество классов
            model_name (str): Название архитектуры
            pretrained (bool): Использовать предобученные веса
            fine_tune_mode (str): Режим fine-tuning:
                'none': только классификатор
                'last': последние слои + классификатор
                'all': все слои (полный fine-tuning)
        """
        super(AircraftClassifier, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.fine_tune_mode = fine_tune_mode
        self.pretrained = pretrained

        print(f"  Создание {model_name} | Классов: {num_classes} | Fine-tune: {fine_tune_mode}")

        # Инициализация модели
        self.backbone, self.num_features = self._initialize_backbone(model_name, pretrained, num_classes)

        # Настройка fine-tuning
        self._setup_fine_tuning(fine_tune_mode)

        # Создание классификатора
        self._create_classifier()

    def _initialize_backbone(self, model_name, pretrained, num_classes):
        """Инициализация базовой архитектуры"""
        if model_name == 'resnet50':
            # ResNet50
            if pretrained:
                weights = ResNet50_Weights.IMAGENET1K_V1
                backbone = models.resnet50(weights=weights)
            else:
                backbone = models.resnet50(weights=None)

            num_features = backbone.fc.in_features

        elif model_name == 'efficientnet':
            # EfficientNet-B0
            if pretrained:
                weights = EfficientNet_B0_Weights.IMAGENET1K_V1
                backbone = models.efficientnet_b0(weights=weights)
            else:
                backbone = models.efficientnet_b0(weights=None)

            num_features = backbone.classifier[1].in_features

        elif model_name == 'mobilenet':
            # MobileNetV2
            if pretrained:
                weights = MobileNet_V2_Weights.IMAGENET1K_V1
                backbone = models.mobilenet_v2(weights=weights)
            else:
                backbone = models.mobilenet_v2(weights=None)

            num_features = backbone.classifier[1].in_features

        elif model_name == 'simple_cnn':
            # Наша простая CNN
            backbone = SimpleCNN(num_classes)
            num_features = None  # Для simple_cnn отдельная обработка

        else:
            raise ValueError(f"Неизвестная архитектура: {model_name}")

        return backbone, num_features

    def _setup_fine_tuning(self, fine_tune_mode):
        """Настройка fine-tuning для разных архитектур"""
        if self.model_name == 'simple_cnn':
            return  # Для простой CNN ничего не делаем

        # Замораживаем/размораживаем слои в зависимости от режима
        if fine_tune_mode == 'none':
            # Только классификатор
            for param in self.backbone.parameters():
                param.requires_grad = False

        elif fine_tune_mode == 'last':
            # Только последние слои + классификатор

            if self.model_name == 'resnet50':
                # Для ResNet: размораживаем layer4 и fc
                for name, param in self.backbone.named_parameters():
                    if 'layer4' not in name and 'fc' not in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

            elif self.model_name in ['efficientnet', 'mobilenet']:
                # Для других моделей: размораживаем последние блоки
                for param in self.backbone.parameters():
                    param.requires_grad = False

                # Размораживаем последние слои
                if self.model_name == 'efficientnet':
                    for param in self.backbone.features[-3:].parameters():
                        param.requires_grad = True
                elif self.model_name == 'mobilenet':
                    for param in self.backbone.features[-5:].parameters():
                        param.requires_grad = True

        elif fine_tune_mode == 'all':
            # Все слои
            for param in self.backbone.parameters():
                param.requires_grad = True

    def _create_classifier(self):
        """Создание улучшенного классификатора"""
        if self.model_name == 'simple_cnn':
            return  # Уже есть свой классификатор

        # Улучшенный классификатор с регуляризацией
        if self.model_name == 'resnet50':
            self.backbone.fc = self._create_enhanced_classifier(self.num_features)

        elif self.model_name == 'efficientnet':
            self.backbone.classifier = self._create_enhanced_classifier(self.num_features)

        elif self.model_name == 'mobilenet':
            self.backbone.classifier = self._create_enhanced_classifier(self.num_features)

    def _create_enhanced_classifier(self, num_features):
        """Создает улучшенный классификатор с регуляризацией"""
        return nn.Sequential(
            nn.Dropout(0.6),  # Высокий dropout для регуляризации

            nn.Linear(num_features, 512),
            nn.LayerNorm(512),  # Заменено на LayerNorm - работает с батчами любого размера
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.LayerNorm(256),  # Заменено на LayerNorm - работает с батчами любого размера
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, self.num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_model_info(self):
        """Возвращает подробную информацию о модели"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Считаем количество обучаемых параметров по слоям
        trainable_by_layer = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_by_layer[name] = param.numel()

        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'pretrained': self.pretrained,
            'fine_tune_mode': self.fine_tune_mode,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'trainable_percent': trainable_params / total_params * 100 if total_params > 0 else 0,
            'trainable_by_layer': trainable_by_layer,
            'frozen_params': total_params - trainable_params
        }


def create_model(model_name='resnet50', num_classes=15, pretrained=True,
                 fine_tune_mode='last', device='cpu'):
    """
    Создает и настраивает модель с правильным fine-tuning

    Args:
        model_name (str): Название архитектуры
        num_classes (int): Количество классов
        pretrained (bool): Использовать предобученные веса
        fine_tune_mode (str): 'none', 'last', или 'all'
        device (str): Устройство для модели

    Returns:
        model: Созданная модель
        model_info: Информация о модели
    """
    print(f"\n{'=' * 50}")
    print(f"СОЗДАНИЕ МОДЕЛИ")
    print(f"{'=' * 50}")

    model = AircraftClassifier(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
        fine_tune_mode=fine_tune_mode
    ).to(device)

    model_info = model.get_model_info()

    # Вывод подробной информации
    print(f"  Архитектура: {model_info['model_name']}")
    print(f"  Количество классов: {model_info['num_classes']}")
    print(f"  Предобученные веса: {'Да' if model_info['pretrained'] else 'Нет'}")
    print(f"  Режим fine-tuning: {model_info['fine_tune_mode']}")
    print(f"  Всего параметров: {model_info['total_params']:,}")
    print(f"  Обучаемых параметров: {model_info['trainable_params']:,} ({model_info['trainable_percent']:.1f}%)")
    print(f"  Заморожено параметров: {model_info['frozen_params']:,}")

    # Показываем топ-5 обучаемых слоев
    if model_info['trainable_by_layer']:
        print(f"\n  Топ-5 обучаемых слоев:")
        sorted_layers = sorted(model_info['trainable_by_layer'].items(),
                               key=lambda x: x[1], reverse=True)[:5]
        for name, params in sorted_layers:
            print(f"    {name}: {params:,}")

    return model, model_info


def test_models():
    """Тестирование всех моделей"""
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ МОДЕЛЕЙ")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    test_cases = [
        {'model_name': 'resnet50', 'fine_tune_mode': 'last'},
        {'model_name': 'efficientnet', 'fine_tune_mode': 'last'},
        {'model_name': 'mobilenet', 'fine_tune_mode': 'last'},
        {'model_name': 'simple_cnn', 'fine_tune_mode': 'all'},
    ]

    for config in test_cases:
        print(f"\nТестируем {config['model_name']} ({config['fine_tune_mode']}):")

        try:
            model, info = create_model(
                model_name=config['model_name'],
                num_classes=15,
                pretrained=True,
                fine_tune_mode=config['fine_tune_mode'],
                device=device
            )

            # Проверяем forward pass
            test_input = torch.randn(2, 3, 224, 224).to(device)
            with torch.no_grad():
                output = model(test_input)

            print(f"  ✓ Вход: {test_input.shape}")
            print(f"  ✓ Выход: {output.shape}")

        except Exception as e:
            print(f"  ✗ Ошибка: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 60)


if __name__ == "__main__":
    test_models()