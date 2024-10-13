import os
import sys

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor


def load_sam2_video_predictor_and_initial_state(
    ckpt_pth: str,
    rgb_jpg_dir: str,
    model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    device: str = "cuda:0",
):
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam2_video_predictor: SAM2VideoPredictor = build_sam2_video_predictor(
        config_file=model_config,
        ckpt_path=ckpt_pth,
        device=device,
    )

    inference_state = sam2_video_predictor.init_state(
        video_path=rgb_jpg_dir,
        offload_video_to_cpu=True,
        async_loading_frames=True,
    )

    return sam2_video_predictor, inference_state


def propagate(
    start_frame_idx: int,
    step: int,
    video_predictor: SAM2VideoPredictor,
    inference_state: dict,
    task_data: dict,
):
    reverse = task_data["reverse"]
    end_idx = start_frame_idx + step if not reverse else start_frame_idx - step
    frames_masks_data = {}

    # add mask to be tracked
    video_predictor.reset_state(inference_state)
    for item in task_data['masks']:
        video_predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=start_frame_idx,
            obj_id=item['original_obj_id'],
            mask=item['segmentation'],
        )

    # propagation
    for out_frame_idx, out_obj_ids, out_masks_logits in video_predictor.propagate_in_video(
            inference_state=inference_state,
            max_frame_num_to_track=step,
            start_frame_idx=start_frame_idx,
            reverse=reverse,
    ):
        if out_frame_idx == start_frame_idx:
            continue
        masks_data = []
        for i in range(len(out_obj_ids)):
            mask = {
                "segmentation": (out_masks_logits[i, 0] > 0.0).cpu().numpy(),
                "original_obj_id": out_obj_ids[i],
                "reverse": reverse,
                "is_key_frame": out_frame_idx == end_idx,
                "frame_idx": out_frame_idx,
                "area": (out_masks_logits[i, 0] > 0.0).sum().cpu().item(),
            }
            masks_data.append(mask)
        frames_masks_data[out_frame_idx] = masks_data

    return frames_masks_data
