import torch

from src.models import create_model


def test_create_model():
    model = create_model(
        num_classes=4,
        pretrained=False,
    )

    x = torch.randn(2, 3, 224, 224)

    y = model(x)

    assert y.shape == (2, 4)
    