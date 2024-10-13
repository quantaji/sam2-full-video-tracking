"""
The basic logic:

1. if there is no mask for this frame then this frame is empty
2. a mask (in the middle) has to be probagated in both direction
3. if there is another mask that propagates to the current mask, then this direction is "used"
    we can recored this by adding "matching from the past/future" list for task at each frame
4. at the begining of the job, we eliminate/merge masks if
    1. there is an existing mask that has intersection over union above the threshold, and 
    2. the direction has already been explored, this is only necessary for the initial task
5, at the end of a job, we eliminate/merge the mask if
    1. there is an existing mask that has intersection over union above the threshold
    2: there is no need at the end of the job to check direction


for each mask it has several attribute
for key frame
1. binary mask segmentation, area
2. original_obj_id
3. left linked original ids (list)
4. right linked original ids (list)
5. whether future or past is explored

for normal frame
1. binary mask
2. original id
3. reverse or not
"""

from pathlib import Path
import numpy as np
import torch
from .utils import IoU


class MaskHandler:

    def __init__(self, mask_dir: str) -> None:
        self.mask_dir = Path(mask_dir)
        self.mask_dir.mkdir(parents=True, exist_ok=True)

    def load_masks(self, frame_idx):
        masks_path = self.mask_dir / "{:06d}.npy".format(frame_idx)
        if masks_path.exists():
            return np.load(str(masks_path), allow_pickle=True).tolist()
        else:
            return []

    def init_key_frame_masks(self, key_frame_idx, sam_1_masks):
        for item in sam_1_masks:
            item.update({
                "linked_future_ids": [],
                "linked_past_ids": [],
                "future_explored": False,
                "past_explored": False,
                "is_key_frame": True,
                "frame_idx": key_frame_idx,
            })
        return sam_1_masks

    def update_key_frame_masks(
        self,
        original_masks: list,
        new_masks: list,
        reverse: bool,
        device: str = "cuda:0",
        disappear_thresh: float = 0.0005,
        iou_thresh: float = 0.8,
    ):
        """
        Given the original mask list and updating mask list, return the new mask list, remaining masks that needs to be propagated, and the correspondance.

        This function is used after propgation
        """
        matched_org_ids, matched_new_ids = [], []
        finished_ids = []
        new_masks_data = []

        org_masks_tsr = torch.from_numpy(
            np.stack(
                [item['segmentation'] for item in original_masks],
                axis=0,
            )).to(device)  # o, h, w
        new_masks_tsr = torch.from_numpy(
            np.stack(
                [item['segmentation'] for item in new_masks],
                axis=0,
            )).to(device)  # n, h, w

        H, W = new_masks_tsr.shape[1:]

        iou = IoU(mask_A=new_masks_tsr, mask_B=org_masks_tsr)  #n, o

        max_iou, matched_indices = iou.max(dim=1)

        for i, new_mask in enumerate(new_masks):
            if max_iou[i].item() > iou_thresh:
                # there is a match
                matched_idx = matched_indices[i].item()
                matched_org_ids.append(
                    original_masks[matched_idx]["original_obj_id"])
                matched_new_ids.append(new_mask["original_obj_id"])
                # also add left or right
                if reverse is not True:
                    original_masks[matched_idx]["linked_past_ids"].append(
                        new_mask["original_obj_id"])
                    original_masks[matched_idx]["past_explored"] = True
                else:
                    original_masks[matched_idx]["linked_future_ids"].append(
                        new_mask["original_obj_id"])
                    original_masks[matched_idx]["future_explored"] = True

                finished_ids.append(new_mask["original_obj_id"])

            elif new_mask["area"] < disappear_thresh * H * W:
                # this mask is too small and disappears
                finished_ids.append(new_mask["original_obj_id"])

            else:
                # this frame is remained
                # the judgement of whether the task is submitted (end or begining of the video) is handled by task handler
                if not reverse:
                    new_mask.update({
                        "linked_past_ids": [new_mask["original_obj_id"]],
                        "linked_future_ids": [],
                        "past_explored":
                        True,
                        "future_explored":
                        False,
                    })
                else:
                    new_mask.update({
                        "linked_past_ids": [],
                        "linked_future_ids": [new_mask["original_obj_id"]],
                        "past_explored":
                        False,
                        "future_explored":
                        True,
                    })
                new_masks_data.append(new_mask)

        updated_masks = original_masks + new_masks_data
        return updated_masks, new_masks_data, matched_org_ids, matched_new_ids, finished_ids

    def filter_query_masks(self, original_masks, task_data):
        """Before a query is actually propagated, there is possibilities that some other masks has been propagated to current frame, and to reduce computation, we do a filtering
        
        no linking or merging happens at this stage, it is all handled by update_key_frame_masks function"""
        #! one can check that the maximum Iou for task_data's mask and original mask is always one
        new_task_mask_data = []
        finished_ids = []

        for task_mask in task_data['masks']:
            matched_mask = next(
                item for item in original_masks
                if item["original_obj_id"] == task_mask["original_obj_id"])
            if (task_data["reverse"] is not True
                    and matched_mask["future_explored"] is not True) or (
                        task_data["reverse"] is True
                        and matched_mask["past_explored"] is not True):
                new_task_mask_data.append(task_mask)
            else:
                finished_ids.append(task_mask["original_obj_id"])

        task_data['masks'] = new_task_mask_data

        return task_data, finished_ids

    def update_normal_frame_masks(
        self,
        original_masks: list,
        new_masks: list,
    ):
        return original_masks + new_masks

    def save_masks(self, frame_idx, masks):
        masks_path = self.mask_dir / "{:06d}".format(frame_idx)
        np.save(str(masks_path), masks)
