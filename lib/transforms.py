import torch

import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode


class RandomResizePad(torch.nn.Module):
    def __init__(
            self,
            scale: tuple,
            fill_value: int = 0,
            padding_mode: str = "constant",
            interpolation: str = InterpolationMode.BILINEAR
    ):
        """
        Resize image to a random smaller size, then pad image with new size to original

        Args:
            scale:
            fill_value:
            padding_mode:
            interpolation:
        """
        super().__init__()
        if not isinstance(scale, tuple):
            raise TypeError("Got inappropriate scale arg")

        if not isinstance(fill_value, int):
            raise TypeError("Got inappropriate fill_value arg")

        if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
            raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")

        self.scale = scale
        self.fill_value = fill_value
        self.padding_mode = padding_mode
        self.interpolation = interpolation

    def get_params(self, img: torch.Tensor):
        """
        Get new small image size and padding size

        Args:
            img: image torch tensor
        """
        w, h = F._get_image_size(img)

        i = torch.randint(
            int(self.scale[0] * 100),
            int(self.scale[1] * 100),
            size=(1,)
        ).item() / 100

        pad_h = int((h - h * i) / 2)
        pad_w = int((w - w * i) / 2)

        resize_h = h - 2 * pad_h
        resize_w = w - 2 * pad_w

        return resize_w, resize_h, pad_w, pad_h

    def forward(self, img: torch.Tensor):
        """
        Get new image

        Args:
            img: image torch tensor
        """
        resize_w, resize_h, pad_w, pad_h = self.get_params(img)

        img = F.resize(img, (resize_w, resize_h), self.interpolation)
        return F.pad(img, (pad_w, pad_h, pad_w, pad_h), self.fill_value, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1}, fill={2}, padding_mode={3})'. \
            format(self.size, self.padding, self.fill, self.padding_mode)
