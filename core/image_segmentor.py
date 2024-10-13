from pathlib import Path
from typing import Union

import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import trange

from .image_reader import ImageReader


def load_sam_auto_gen(
    ckpt_pth: str,
    model_type: str = "vit_h",
    points_per_side: int = 64,
    device: str = "cuda:0",
    pred_iou_thresh: float = 0.8,
):
    sam = sam_model_registry[model_type](checkpoint=ckpt_pth).to(device)
    sam_auto_gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
    )
    return sam_auto_gen


def segment_with_sam(
    rgb_dir: str,
    save_dir: Union[str, Path],
    sam_auto_gen: SamAutomaticMaskGenerator,
    min_size: int = -1,
    step: int = 5,
    max_masks_per_frame: int = 144,
):
    img_reader = ImageReader(
        rgb_dir=rgb_dir,
        min_size=min_size,
    )
    num_images = len(img_reader)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx in trange(0, num_images, step):
        image_arr = img_reader.get_single_image(idx=frame_idx)
        mask_data = sam_auto_gen.generate(image_arr)

        num_masks = len(mask_data)

        keep_list = np.ones(shape=(num_masks, ), dtype=bool)

        if num_masks > max_masks_per_frame:
            area_arr = np.array([item['area'] for item in mask_data])
            keep_list = np.argsort(-area_arr) < max_masks_per_frame

        save_data = []
        obj_id_iter = 1 + frame_idx * max_masks_per_frame
        for i in range(num_masks):
            if keep_list[i]:
                item = mask_data[i]
                item.update({
                    "is_key_frame": True,
                    "original_obj_id": obj_id_iter,
                })
                obj_id_iter += 1
                save_data.append(item)

        save_path = str(save_dir / "{:06d}.npy".format(frame_idx))
        np.save(save_path, save_data)
