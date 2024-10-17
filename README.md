# sam2-full-video-tracking

This repository is a tool to automatically over-segment and track a given video. It tries to generate overlapping masks of different granularity. This might be useful for you if you
1. need to extract masks that have sub-semantic granularity, e.g. want the handle of a cup to be also detected, and
2. expecting each pixel can belong to several different masks that are overlapping.

Brief explaination of wha this pipeline does:
1. split a video into clips,
2. run over-segmentation with SAM1 on the beginning frames of each clip, as SAM1 generates more masks
3. use SAM2 to propagate each masks to neighboring key frames.
4. Assotiate propagated masks to existing masks

A dropback of this pipeline is that, there might be some small or isolated masks that can be propagated long enough in the whole video. Over 70% of the time this pipeline is dealing with this tail-distribution case, in order to segment the video as complete as possible. This example video tasks about 4 hours to fully segment. However, you can adjust the hyper-parameter to make it stop earlier.

| RGB-input                                                                                     | SAM2 over-segmentation and tracking                                                           |
| --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| <video src="https://github.com/user-attachments/assets/3485a8c5-f120-416d-bf26-bccfbd8fe633"> | <video src="https://github.com/user-attachments/assets/5ae69048-3bb8-4f25-bc99-b6a31e836f3f"> |


## Installation
We assume you have a cuda device and conda installed as your package manager. To install the environment run
```sh
bash install_env.sh
```
and it will create a conda environment called sam2.

## Usage
Please check the [demo notbook](demo/demo.ipynb) for usage! 

Many thanks for checking out this repo! 😃
