from src.models import (
    create_model,
    get_optimizer,
)


def test_get_optimizer():
    model = create_model(
        num_classes=4,
        pretrained=False,
    )

    optimizer = get_optimizer(
        model,
        optimizer_name="adam",
    )

    assert optimizer is not None
    