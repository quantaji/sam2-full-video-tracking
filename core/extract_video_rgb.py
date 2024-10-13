from pathlib import Path
from typing import  Union

import imageio
import numpy as np
from PIL import Image

from tqdm import trange


def extract_video_rgb(
    video_path: Union[str, Path],
    save_workspace_dir: Union[str, Path],
):
    save_workspace_dir = Path(save_workspace_dir)
    save_workspace_dir.mkdir(parents=True, exist_ok=True)

    reader = imageio.get_reader(str(video_path))

    rgb_save_dir = save_workspace_dir / "data" / "rgb"
    rgb_save_dir.mkdir(parents=True, exist_ok=True)

    n_frames = reader.count_frames()

    for i in trange(n_frames):
        img = np.array(reader.get_next_data())
        Image.fromarray(img).save(str(rgb_save_dir / "{:06d}.jpg".format(i)))
