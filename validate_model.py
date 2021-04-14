from torchvision import transforms

from utils import get_dataset
from utils import get_loader

from lib.utils import load_checkpoint

# TRANSFORMS
transforms_pipeline = transforms.Compose(
    [
        transforms.Resize((220, 220)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ),
    ]
)


if __name__ == '__main__':
    pass
