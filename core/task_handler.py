"""A queue task file is a dict, it should contain
1. start_frame_id
2. reverse of not
3. mask information
    1. binary
    2. original id

save format: {task_id}.npy


for a union find, it is a dictionary that map the original id to the current id

also monitoring after each merge
1. how many object id exists
2. how many object id terminated


before a task is launched, it will first
1. check existing mask, if there is already an intersection, then remove the intersected mask
2. propagate the rest mask
3. merge the mask at final frame
4. for unmatched masks, push it to the queue
    keep the direction unchanged

for this task handler, it first
submit all forward task
submit all reverse task
"""
import json
from pathlib import Path

import numpy as np
from sam2.sam2_video_predictor import SAM2VideoPredictor
from tqdm import tqdm, trange

from .mask_handler import MaskHandler
from .propagator import propagate


class Task:

    def __init__(
        self,
        task_id: int = 0,
        start_frame: int = None,
        reverse: bool = False,
        masks: list = None,
        queue_dir: str = None,
        data_pth: str = None,
    ) -> None:

        if data_pth is not None:
            self.task_data = np.load(data_pth, allow_pickle=True).item()
            if isinstance(self.task_data['masks'], np.ndarray):
                self.task_data['masks'] = self.task_data['masks'].tolist()
            self.task_id = self.task_data["task_id"]
            self.path = Path(data_pth)

        else:
            self.task_id = task_id
            self.task_data = {
                "task_id": task_id,
                "start_frame": start_frame,
                "reverse": reverse,
                "masks": masks,
            }
            self.path = Path(queue_dir) / "{:06d}.npy".format(self.task_id)

    def save(self):
        np.save(str(self.path), self.task_data)

    def delete(self):
        self.path.unlink(missing_ok=True)


class TaskHandler:

    def __init__(
        self,
        queue_dir: str,
        sam_mask_dir: str,
        save_mask_dir: str,
        video_predictor: SAM2VideoPredictor,
        inference_state: dict,
        step: int = 5,
        device: str = "cuda:0",
        disappear_thresh: float = 0.0008,
        iou_thresh: float = 0.7,
    ) -> None:

        self.step = step
        self.sam_mask_dir = Path(sam_mask_dir)
        self.id_func = lambda x: int(x.stem)
        self.sam_mask_list = sorted(
            list(self.sam_mask_dir.glob("*.npy")),
            key=self.id_func,
        )
        self.key_frame_list = [self.id_func(p) for p in self.sam_mask_list]

        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(exist_ok=True, parents=True)

        self.video_predictor = video_predictor
        self.inference_state = inference_state

        self.device = device
        self.disappear_thresh = disappear_thresh
        self.iou_thresh = iou_thresh

        # mask handler
        self.mask_handler = MaskHandler(mask_dir=save_mask_dir)
        self.save_mask_dir = Path(save_mask_dir)

        self.num_frames = inference_state['num_frames']

        # counter
        self.task_counter = 0
        self.terminated_mask = []

        # original id to merged id mapping
        self.id_map = {}

        # flags
        self.initialized: bool = False

        # ! TODO: add resume function after the first demo

    def submit_initial_tasks(self):
        # forward
        for path, key_frame_idx in tqdm(
                zip(
                    self.sam_mask_list,
                    self.key_frame_list,
                )):

            mask_data = np.load(str(path), allow_pickle=True)

            # save the key frame mask
            self.mask_handler.save_masks(
                frame_idx=key_frame_idx,
                masks=self.mask_handler.init_key_frame_masks(
                    key_frame_idx=key_frame_idx,
                    sam_1_masks=mask_data,
                ))

            # initialize the id map
            for item in mask_data:
                self.id_map[int(item["original_obj_id"])] = int(
                    item["original_obj_id"])

            # save
            if key_frame_idx == self.num_frames - 1:
                continue
            task = Task(
                task_id=self.task_counter,
                start_frame=key_frame_idx,
                reverse=False,
                masks=mask_data,
                queue_dir=str(self.queue_dir),
            )
            task.save()
            self.task_counter += 1

        # backward
        for path, key_frame_idx in tqdm(
                zip(
                    self.sam_mask_list,
                    self.key_frame_list,
                )):
            if key_frame_idx == 0:
                continue
            mask_data = np.load(str(path), allow_pickle=True)

            # save task to queue
            task = Task(
                task_id=self.task_counter,
                start_frame=key_frame_idx,
                reverse=True,
                masks=mask_data,
                queue_dir=str(self.queue_dir),
            )
            task.save()
            self.task_counter += 1

        self.save_id_map()

        self.initialized = True

    def save_id_map(self):
        with open(str(self.queue_dir / "id_map.json"), "w") as outfile:
            json.dump(self.id_map, outfile, indent=4)

    def load_id_map(self):
        with open(str(self.queue_dir / "id_map.json"), "r") as infile:
            self.id_map = json.load(infile)

    def load_task_count(self):
        task_id_list = [int(p.stem) for p in self.queue_dir.glob('*.npy')]
        self.task_counter = max(task_id_list) + 1

    def get_queue_top_task(self):
        assert self.initialized
        task_data_list = sorted(
            list(self.queue_dir.glob("*.npy")),
            key=self.id_func,
        )

        if len(task_data_list) == 0:
            return None

        task_data_path = task_data_list[0]
        return Task(data_pth=str(task_data_path))

    def run_one_task(self):
        # Step 0 load the task information
        task = self.get_queue_top_task()
        if task is None:
            print("No existing tasks, all job finished")
            return False

        print(f"running task id: {task.task_data['task_id']}")
        start_frame_idx = task.task_data['start_frame']
        reverse = task.task_data['reverse']
        end_frame_idx = start_frame_idx + self.step if not reverse else start_frame_idx - self.step

        finished_ids = []

        # Step 1 load the existing mask and remove the already matched mask save the new matching information
        start_frame_masks = self.mask_handler.load_masks(start_frame_idx)
        filterd_task_data, tmp_finished_ids = self.mask_handler.filter_query_masks(
            original_masks=start_frame_masks,
            task_data=task.task_data,
        )
        finished_ids += tmp_finished_ids
        if len(filterd_task_data['masks']) == 0:
            # no more masks needs to be tracked
            print("no masks need to track for this task! Continue!\n")
            task.delete()
            return True

        # Step 2 propagation
        frames_masks_data = propagate(
            start_frame_idx=start_frame_idx,
            step=self.step,
            video_predictor=self.video_predictor,
            inference_state=self.inference_state,
            task_data=filterd_task_data,
        )

        # Step 3 load the end frame's
        end_frame_out_of_range = end_frame_idx < 0 or end_frame_idx >= self.num_frames
        if not end_frame_out_of_range:
            end_frame_masks = self.mask_handler.load_masks(end_frame_idx)

        # step 4: save, we save here to make the operation as atomic as possible
        ## first for all mask in start frame
        propagated_ids = set(
            [item['original_obj_id'] for item in filterd_task_data['masks']])
        for mask in start_frame_masks:
            if mask['original_obj_id'] in propagated_ids:
                if not reversed:
                    mask['future_explored'] = True
                else:
                    mask["past_explored"] = True
        self.mask_handler.save_masks(
            frame_idx=start_frame_idx,
            masks=start_frame_masks,
        )

        ## then all the intermediate frame
        for frame_idx in frames_masks_data.keys():
            if frame_idx in {start_frame_idx, end_frame_idx}:
                continue
            frame_masks = self.mask_handler.load_masks(frame_idx)
            frame_masks = self.mask_handler.update_normal_frame_masks(
                original_masks=frame_masks,
                new_masks=frames_masks_data[frame_idx],
            )
            self.mask_handler.save_masks(
                frame_idx=frame_idx,
                masks=frame_masks,
            )

        # then the end frame
        if not end_frame_out_of_range:
            (end_frame_masks, new_masks, matched_org_ids, matched_new_ids,
             tmp_finished_ids) = self.mask_handler.update_key_frame_masks(
                 original_masks=end_frame_masks,
                 new_masks=frames_masks_data[end_frame_idx],
                 reverse=reverse,
                 device=self.device,
                 disappear_thresh=self.disappear_thresh,
                 iou_thresh=self.iou_thresh,
             )
            self.mask_handler.save_masks(
                frame_idx=end_frame_idx,
                masks=end_frame_masks,
            )
            finished_ids += tmp_finished_ids

            ## step 5: update object mapping
            for org_id, new_id in zip(matched_org_ids, matched_new_ids):
                self.id_map[new_id] = org_id

            self.save_id_map()

            # step 6: submit new task and delete current task
            if len(new_masks) > 0:
                new_task = Task(
                    task_id=self.task_counter,
                    start_frame=end_frame_idx,
                    reverse=reverse,
                    masks=new_masks,
                    queue_dir=str(self.queue_dir),
                )
                new_task.save()
                self.task_counter += 1

        task.delete()
        self.terminated_mask += finished_ids

        # get stats
        print(f"Task with ID: {task.task_data['task_id']} is finished!")
        print(
            f"{len(filterd_task_data['masks'])} is tracked during this task.")
        if not end_frame_out_of_range:
            print(f"{len(matched_new_ids)} masks are matched.")
        print(
            f"{len(finished_ids)} masks are terminated during this task. In total, {len(self.terminated_mask)}/{len(self.id_map.keys())} masks are terminated."
        )
        if not end_frame_out_of_range and len(new_masks) > 0:
            print(f"A new task of {len(new_masks)} masks is submitted.")
        print('')

        return True
