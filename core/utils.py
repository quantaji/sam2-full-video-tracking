from typing import List, Union, Dict

import numpy as np
import torch


class ID2RGBConverter:

    def __init__(self):
        self.all_id = []
        self.obj_to_id = {0: 0}

    def _id_to_rgb(self, id: int):
        rgb = np.zeros((3, ), dtype=np.uint8)
        for i in range(3):
            rgb[i] = id % 256
            id = id // 256
        return rgb

    def convert(self, obj: int):
        if obj in self.obj_to_id:
            id = self.obj_to_id[obj]
        else:
            while True:
                id = np.random.randint(255, 256**3)
                if id not in self.all_id:
                    break
            self.obj_to_id[obj] = id
            self.all_id.append(id)

        return id, self._id_to_rgb(id)


def IoU(mask_A: torch.Tensor, mask_B: torch.Tensor):
    """Boolean mask of shape n_mask, h, w
    to ensure the matching algorithm is working, we also add a pesudo label with iou=1e-8"""
    intersection = torch.einsum("ijk,ljk->il", mask_A.float(), mask_B.float())
    area_A = mask_A.count_nonzero(dim=[1, 2]).reshape(-1, 1)
    area_B = mask_B.count_nonzero(dim=[1, 2]).reshape(1, -1)
    IoU = intersection / (area_A + area_B - intersection + 1e-8)

    return IoU


def flatten_mask(
    mask: Union[np.ndarray, List[Dict], List[np.ndarray]],
    object_id_list=None,
    device: str = "cuda:0",
):

    if isinstance(mask, list):
        if isinstance(mask[0], np.ndarray):
            mask = np.stack(mask, axis=0)
        elif isinstance(mask[0], Dict):
            assert "segmentation" in mask[0].keys()
            mask = np.stack([item['segmentation'] for item in mask], axis=0)
        else:
            raise NotImplementedError
    elif isinstance(mask, np.ndarray):
        pass
    else:
        raise NotImplementedError

    mask_tsr = torch.from_numpy(mask).to(device)
    areas = mask_tsr.flatten(-2).sum(-1)
    scores = (areas.max() * 2 - areas).unsqueeze(-1).unsqueeze(-1)
    scored_masks = mask_tsr.float() * scores
    scored_masks_with_bg = torch.cat(
        [
            torch.zeros(
                (1, *mask_tsr.shape[1:]), device=device) + 0.1, scored_masks
        ],
        dim=0,
    )
    output_mask = torch.argmax(scored_masks_with_bg, dim=0)

    if object_id_list is not None:
        id_map = torch.tensor([
            0,
        ] + object_id_list, device=device)
        output_mask = id_map[output_mask]

    return output_mask.cpu().numpy()


def viz_mask(
    flattened_mask: np.ndarray,
    converter: ID2RGBConverter,
    alpha: Dict = None,
):
    H, W = flattened_mask.shape
    annotated_rgb = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    for obj_id in np.unique(flattened_mask):
        color = converter.convert(obj_id)[1]
        if alpha is not None:
            color = (color.astype(float) * alpha[int(obj_id)]).astype(np.uint8)
        annotated_rgb[flattened_mask == obj_id] = color
    return annotated_rgb
