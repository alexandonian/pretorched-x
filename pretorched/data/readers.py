import logging
import subprocess
from collections import namedtuple

import numpy as np

from pathlib import Path
from typing import Union, List, Iterator, IO

import cv2
from PIL import Image

from torchvideo.samplers import frame_idx_to_list

_LOG = logging.getLogger(__name__)

VideoInfo = namedtuple("VideoInfo", ("height", "width", "n_frames"))
_VIDEO_FILE_EXTENSIONS = {
    "mp4",
    "webm",
    "avi",
    "3gp",
    "wmv",
    "mpg",
    "mpeg",
    "mov",
    "mkv",
}


def lintel_loader(
    file: Union[str, Path, IO[bytes]], frames_idx: Union[slice, List[slice], List[int]]
) -> Iterator[Image.Image]:
    import lintel

    if isinstance(file, str):
        file = Path(file)
    if isinstance(file, Path):
        _LOG.debug("Loading data from {}".format(file))
        with file.open("rb") as f:
            video = f.read()
    else:
        video = file.read()

    frames_idx = np.array(frame_idx_to_list(frames_idx))
    assert isinstance(frames_idx, np.ndarray)
    load_idx, reconstruction_idx = np.unique(frames_idx, return_inverse=True)
    _LOG.debug("Converted frames_idx {} to load_idx {}".format(frames_idx, load_idx))
    frames_data, width, height = lintel.loadvid_frame_nums(
        video, frame_nums=load_idx, should_seek=False
    )
    frames = np.frombuffer(frames_data, dtype=np.uint8)
    # TODO: Support 1 channel grayscale video
    frames = np.reshape(frames, newshape=(len(load_idx), height, width, 3))
    frames = frames[reconstruction_idx]
    return (Image.fromarray(frame) for frame in frames)


def cv2_loader(file, frame_idx=None, fps=30):
    def read_frames(file, frame_idx):
        # Open video file
        video_capture = cv2.VideoCapture(file)
        # video_capture.set(cv2.CAP_PROP_FPS, fps)

        count = 0
        while video_capture.isOpened():
            # Grab a single frame of video
            ret = video_capture.grab()

            # Bail out when the video file ends
            if not ret:
                break
            if count in frame_idx:
                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                ret, frame = video_capture.retrieve()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame
            count += 1

    return (Image.fromarray(frame) for frame in read_frames(file, frame_idx))


def default_loader(
    file: Union[str, Path, IO[bytes]], frames_idx: Union[slice, List[slice], List[int]]
) -> Iterator[Image.Image]:
    from torchvideo import get_video_backend

    backend = get_video_backend()
    if backend == "lintel":
        loader = lintel_loader
    elif backend == 'cv2':
        loader = cv2_loader
    else:
        raise ValueError("Unknown backend '{}'".format(backend))
    return loader(file, frames_idx)


def _get_videofile_frame_count(video_file_path: Path) -> int:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_file_path),
    ]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )
    # Final character of output is a newline so we drop it
    n_frames = int(result.stdout.decode("utf-8").split("\n")[0])
    return n_frames


def _is_video_file(path: Path) -> bool:
    extension = path.name.lower().split(".")[-1]
    return extension in _VIDEO_FILE_EXTENSIONS
