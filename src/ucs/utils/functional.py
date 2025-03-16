import torch
import torch.nn.functional as F


def one_hot(
    target: torch.Tensor, num_classes: int, ignore_index: int = 255, argmax=True
):
    """
    Converts target mask into one-hot encoding while handling ignore_index separately.
    """
    assert ignore_index >= num_classes, "ignore index need to be >= num_classes"
    if target.dim() == 4 and argmax:
        target = torch.argmax(target, dim=1)

    if ignore_index is None:
        return F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2)

    ignore_mask = target == ignore_index
    target = target.clone()
    target[ignore_mask] = num_classes  # Assign ignored pixels to new last class

    encoded = F.one_hot(target, num_classes=num_classes + 1).permute(0, 3, 1, 2)

    # 🔹 Remove the ignored class before returning
    return encoded[:, :-1, :, :]  # Remove ignored class
