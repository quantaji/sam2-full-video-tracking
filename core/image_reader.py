from PIL import Image
from pathlib import Path
from typing import Callable
import torch
import numpy as np


class ImageReader:

    def __init__(
            self,
            rgb_dir: str,
            min_size: int = -1,
            glob_str: str = "*.jpg",
            idx_func: Callable = lambda x: int(x.stem),
    ) -> None:
        self.image_path = Path(rgb_dir)

        self.image_path_list = sorted(
            list(self.image_path.glob(glob_str)),
            key=idx_func,
        )
        assert len(self.image_path_list) > 0

        if min_size == -1:
            self.need_resize = False
            self.H, self.W = None, None
        else:
            self.need_resize = True
            prob_image_arr = np.array(
                Image.open(self.image_path_list[0]).convert("RGB"))
            h, w = prob_image_arr.shape[:2]
            scale = min_size / min(h, w)
            self.H, self.W = int(h * scale), int(w * scale)

    def __len__(self):
        return len(self.image_path_list)

    def get_single_image(self, idx, dtype="numpy", device="cpu"):
        """
        return a image array or tensor of shape H, W, 3
        """
        assert dtype in ["numpy", "torch"]
        image = Image.open(self.image_path_list[idx]).convert("RGB")
        if self.need_resize:
            image = image.resize(size=(self.W, self.H))
        image_arr = np.array(image)

        if dtype == "torch":
            return torch.from_numpy(image_arr).to("device")
        else:
            return image_arr

    def get_image_batch(
        self,
        start_idx,
        end_idx,
        step=1,
        order="ascending",
        dtype="numpy",
        device="cpu",
    ):
        assert dtype in ["numpy", "torch"]
        assert order in ['ascending', "descending"]

        idx_list = list(range(start_idx, end_idx, step))
        if order == "descending":
            idx_list = idx_list[::-1]

        image_arr_list = []
        for idx in idx_list:
            image = Image.open(self.image_path_list[idx]).convert("RGB")
            if self.need_resize:
                image = image.resize(size=(self.W, self.H))
            image_arr = np.array(image)
            image_arr_list.append(image_arr)
        image_arr_list = np.stack(image_arr_list, axis=0)  # N, H, W, 3

        if dtype == "torch":
            return torch.from_numpy(image_arr_list).to("device")
        else:
            return image_arr_list
