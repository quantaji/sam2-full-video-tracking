default_config = {
    "disappear_threshold":
    0.0005,  # if a mask is smaller than this ratio, it is not save at the first stage, and after tracking it will be recognized as disappear and stop tracking
    "iou_threshold": 0.8,  # if two mask iou > 0.8 then they are matched
    "step": 5,  # the length of tracking [0], 1, 2, 3, 4, [5],
    "max_masks_per_frame": 144,  # maximum number of masks after sam_1, 
    # "propagation_batch_size": 32, # number of mask that is propagated together, this is the maximum number of masks for a single task in a queue
    # "iou_batch_size": 64, # batch_size of calculating iou
    "min_size": -1,  # the pixel size of min edge, -1 means unchagned
    "sam1_points_per_side": 64,
    "sam1_pred_iou_thrshold": 0.8,
}
