"""
MobileNetV3 Model Definition for Gastric Cancer Classification
Supports both MobileNetV3-Small and MobileNetV3-Large variants.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Dict, Any


class MobileNetV3Classifier(nn.Module):
    """
    MobileNetV3-based classifier for gastric cancer classification
    Supports 8 classes: TUM, STR, NOR, MUS, MUC, LYM, DEB, ADI
    """

    def __init__(
        self,
        num_classes: int = 8,
        variant: str = "small",
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ):
        """
        Initialize MobileNetV3 classifier

        Args:
            num_classes: Number of output classes (default: 8)
            variant: Model variant ("small" or "large")
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for the classifier head
        """
        super(MobileNetV3Classifier, self).__init__()

        self.num_classes = num_classes
        self.variant = variant

        # Load pretrained MobileNetV3 variant correctly
        if variant == "small":
            base_model = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
                if pretrained else None
            )
            in_features = 576  # Output feature size for MobileNetV3-Small
        elif variant == "large":
            base_model = models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
                if pretrained else None
            )
            in_features = 1280  # Output feature size for MobileNetV3-Large
        else:
            raise ValueError(f"Unknown variant: {variant}. Choose 'small' or 'large'.")

        # Replace classifier head
        base_model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
            nn.LogSoftmax(dim=1)
        )

        # Store the model
        self.backbone = base_model

        # Initialize the classifier weights
        self._initialize_classifier_weights()

    def _initialize_classifier_weights(self):
        """Initialize classifier weights using Xavier initialization."""
        for m in self.backbone.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass."""
        return self.backbone(x)

    def get_features(self, x):
        """Extract features before the final classifier."""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)

    def freeze_backbone(self):
        """Freeze the feature extractor for transfer learning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze the feature extractor for fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True


def get_model(
    num_classes: int = 8,
    variant: str = "small",
    pretrained: bool = True,
    **kwargs
) -> MobileNetV3Classifier:
    """
    Factory function to create a MobileNetV3 model.
    """
    return MobileNetV3Classifier(
        num_classes=num_classes,
        variant=variant,
        pretrained=pretrained,
        **kwargs,
    )


def get_model_info(model: MobileNetV3Classifier) -> Dict[str, Any]:
    """Return model info and statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "variant": model.variant,
        "num_classes": model.num_classes,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),
    }


# For testing
if __name__ == "__main__":
    model = get_model(num_classes=8, variant="small", pretrained=True)
    print("✅ Model created successfully!")
    info = get_model_info(model)
    print(f"Model Info: {info}")
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
