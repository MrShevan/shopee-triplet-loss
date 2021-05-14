import os
import logging
import textwrap
import numpy as np

from PIL import Image, ImageOps, ImageFont, ImageDraw
from RandomWordGenerator import RandomWord

import torch

import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode


class RandomText(torch.nn.Module):
    def __init__(
        self,
        fonts_dir: str,
        p: float = 0.5,
        max_word_size: int = 15,
        max_num_of_words: int = 3,
        max_row_chars: int = 20,
        constant_word_size: bool = False,
        fill_color: str = "#000000",  # Black
        font_size: int = None,
    ):
        """

        Args:
            fonts_dir:
            p:
            max_word_size:
            max_num_of_words:
            max_row_chars:
            constant_word_size:
            fill_color:
            font_size:
        """
        super().__init__()
        self.fonts_dir = fonts_dir
        self.p = p
        self.max_word_size = max_word_size
        self.max_num_of_words = max_num_of_words
        self.max_row_chars = max_row_chars
        self.constant_word_size = constant_word_size
        self.fill_color = fill_color
        self.font_size = font_size
        self.fonts_path = os.listdir(self.fonts_dir)

        self.word_generator = RandomWord(
            max_word_size=max_word_size,
            constant_word_size=constant_word_size
        )

    @staticmethod
    def _get_text_position(img, text, font):
        rows = {i: len(line) for i, line in enumerate(text)}
        max_row = sorted(rows.items(), key=lambda x: x[1], reverse=True)[0][0]

        img_width, img_height = img.size
        font_width, font_height = font.getsize(text[max_row])

        max_width = img_width - font_width
        max_height = img_height - font_height * len(rows)

        it_w = np.random.choice([0, 1, 2, 3])  # max_width // 3
        it_h = np.random.choice([0, 3])  # max_height // 4

        w_low, w_high = int(it_w * max_width / 4), int((it_w + 1) * max_width / 4)
        h_low, h_high = int(it_h * max_height / 4), int((it_h + 1) * max_height / 4)

        if w_high > w_low:
            w = np.random.randint(low=w_low, high=w_high)
        else:
            w = w_low

        if h_high > h_low:
            h = np.random.randint(low=h_low, high=h_high)
        else:
            h = h_low

        return w, h

    def forward(self, img):
        if torch.rand(1) < self.p:
            try:
                draw = ImageDraw.Draw(img)

                font_size = self.font_size
                if not font_size:
                    font_size = np.random.choice([20, 35, 50])

                font_path = np.random.choice(self.fonts_path)
                font = ImageFont.truetype(
                    font=os.path.join(self.fonts_dir, font_path),
                    size=font_size
                )

                num_of_words = np.random.randint(low=1, high=(self.max_num_of_words + 1))
                words = self.word_generator.getList(num_of_words=num_of_words)

                text = textwrap.wrap(' '.join(words).lower(), width=self.max_row_chars)

                w, h = self._get_text_position(img, text, font)

                y_text = h
                for line in text:
                    width, height = font.getsize(line)
                    draw.text(
                        xy=(w, y_text),
                        text=line,
                        font=font,
                        fill=self.fill_color
                    )
                    y_text += height + 4

            except Exception as msg:
                logging.error(msg)

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomBound(torch.nn.Module):
    def __init__(self, p: float = 0.5):
        """

        Args:
            p:
        """
        super().__init__()
        self.p = p
        self.colors_dict = {
            "Red": "#f00",
            "Electric Blue": "#7DF9FF",
            "Cadet Blue": "#5F9EA0",
            "Egyptian Blue": "#1434A4",
            "Jade": "#00A36C",
            "Iris": "#5D3FD3",
            "Seafoam Green": "#9FE2BF",
            "Brown": "#A52A2A",
            "Bronze": "#CD7F32",
            "Burnt Sienna": "#E97451",
            "Coffee": "#6F4E37",
            "Dark Brown": "#5C4033",
            "Dark Red": "#8B0000",
            "Khaki": "#F0E68C",
            "Olive Green": "#808000",
            "Aqua": "#00FFFF",
            "Aquamarine": "#7FFFD4",
            "Lime Green": "#32CD32",
            "Bright Green": "#AAFF00",
            "Citrine": "#E4D00A",
            "Coral Pink": "#F88379",
            "Neon Orange": "#FF5F1F",
            "Crimson": "#DC143C",
            "Fuchsia": "#FF00FF",
            "Pink": "#FFC0CB",
            "Neon Red": "#FF0000",
            "Bright Yellow": "#FFEA00",
            "Gold": "#FFD700",
            "Black": "#000000",
            "Licorice": "#1B1212"
        }

    def forward(self, img):
        if torch.rand(1) < self.p:
            try:
                border = np.random.choice([5, 10, 20, 40])
                color = np.random.choice(list(self.colors_dict.keys()))

                img = ImageOps.expand(img, border=border, fill=self.colors_dict[color])

            except Exception as msg:
                logging.error(msg)

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomWatermark(torch.nn.Module):
    def __init__(self, watermarks_dir: str, p: float = 0.5):
        """


        Args:
            p: probability of the image being flipped. Default value is 0.5
            watermarks_dir: directory with watermark images
        """
        super().__init__()
        self.watermarks_dir = watermarks_dir
        self.watermarks_path = os.listdir(self.watermarks_dir)
        self.p = p

    @staticmethod
    def _get_watermark_position(img, watermark):
        """


        Args:
            img:  image torch tensor or PIL image
            watermark:  image torch tensor or PIL image
        """
        width, height = img.size
        watermark_width, watermark_height = watermark.size

        max_position_w = width - watermark_width
        max_position_h = height - watermark_height

        center_w = max_position_w // 2
        center_h = max_position_h // 2

        position_x = center_w
        position_y = center_h
        while position_x == center_w and position_y == center_h:
            position_x = np.random.choice([0, center_w, max_position_w])
            position_y = np.random.choice([0, center_h, max_position_h])

        return position_x, position_y

    def forward(self, img):
        """
        Get new image

        Args:
            img: image torch tensor or PIL image
        """
        if torch.rand(1) < self.p:
            try:
                watermark_path = np.random.choice(self.watermarks_path)
                watermark = Image.open(os.path.join(self.watermarks_dir, watermark_path))

                position = self._get_watermark_position(img, watermark)

                img.paste(watermark, position, mask=watermark)

            except Exception as msg:
                logging.error(msg)

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(watermarks_dir={0}, p={1})'. \
            format(self.watermarks_dir, self.p)


class RandomResizeWithPad(torch.nn.Module):
    def __init__(
            self,
            scale: tuple,
            p: float = 0.5,
            fill_value: int = 256,
            padding_mode: str = "constant",
            interpolation: str = InterpolationMode.BILINEAR
    ):
        """
        Resize image to a random smaller size, then pad image with new size to original.
        Tip: Good practise resize image to smaller before this transform.

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
        self.p = p
        self.fill_value = fill_value
        self.padding_mode = padding_mode
        self.interpolation = interpolation

    def _get_params(self, img):
        """
        Get new image size and padding size

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

    def forward(self, img):
        """
        Get new image

        Args:
            img: image torch tensor or PIL image
        """
        if torch.rand(1) < self.p:
            resize_w, resize_h, pad_w, pad_h = self._get_params(img)

            img = F.resize(img, (resize_w, resize_h), self.interpolation)
            return F.pad(img, (pad_w, pad_h, pad_w, pad_h), self.fill_value, self.padding_mode)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1}, fill={2}, padding_mode={3})'. \
            format(self.size, self.padding, self.fill, self.padding_mode)
