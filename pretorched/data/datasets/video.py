import numbers
import io
import os
from pathlib import Path
import zipfile
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch.utils.data
from PIL.Image import Image

from torchvideo.datasets import RecordSet, VideoDataset
from torchvideo.internal.readers import (_get_videofile_frame_count,
                                         _is_video_file)
# from torchvideo.internal.utils import frame_idx_to_list
from torchvideo.samplers import FrameSampler, FullVideoSampler
from torchvideo.transforms import PILVideoToTensor
from .record_set import Label, LabelSet


Transform = Callable[[Any], torch.Tensor]
PILVideoTransform = Callable[[Iterator[Image]], torch.Tensor]
NDArrayVideoTransform = Callable[[np.ndarray], torch.Tensor]


_default_sampler = FullVideoSampler


class VideoRecordDataset(VideoDataset):

    def __init__(
        self,
        root: Union[str, Path],
        record_set: RecordSet,
        sampler: FrameSampler = _default_sampler(),
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        frame_counter: Optional[Callable[[Path], int]] = None,
    ) -> None:

        self.root = root
        self.sampler = sampler
        self.record_set = record_set

        if frame_counter is None:
            frame_counter = _get_videofile_frame_count
        self.frame_counter = frame_counter

        if transform is None:
            transform = PILVideoToTensor()
        self.transform = transform

        if target_transform is None:
            target_transform = int
        self.target_transform = target_transform
        self.video_lens = {}

    def __getitem__(self, index: int) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        record = self.record_set[index]
        video_path = os.path.join(self.root, record.path)
        video_length = record.num_frames or self.frame_counter(video_path)
        # try:
        # video_length = self.video_lens[index]
        # except KeyError:
        # video_length = record.num_frames or self.frame_counter(video_path)
        # self.video_lens[index] = video_length
        frame_inds = self.sampler.sample(video_length)
        frames = self._load_frames(video_path, frame_inds)
        label = record.label

        if self.transform is not None:
            frames = self.transform(frames)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return frames, label

    def __len__(self):
        return len(self.record_set)


class VideoRecordZipDataset(VideoDataset):

    def __init__(
        self,
        root,
        record_set: RecordSet,
        sampler: FrameSampler = _default_sampler(),
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        frame_counter: Optional[Callable[[Path], int]] = None,
    ) -> None:

        self.root = root
        self.zipfilenames = [z for z in os.listdir(self.root) if z.endswith('.zip')]
        self.zips = {}
        for zfname in self.zipfilenames:
            cat = zfname.rstrip('.zip')
            self.zips[cat] = zipfile.ZipFile(os.path.join(root, zfname))
            # self.zips[cat] = zip_file

        self.sampler = sampler
        self.record_set = record_set

        if frame_counter is None:
            frame_counter = StaticFrameCounter()
        self.frame_counter = frame_counter

        if transform is None:
            transform = PILVideoToTensor()
        self.transform = transform

        if target_transform is None:
            target_transform = int
        self.target_transform = target_transform

    def __getitem__(self, index: int) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        record = self.record_set[index]
        video_path = record.path
        category = os.path.dirname(video_path)
        video_path = io.BytesIO(self.zips[category].read(video_path))
        video_length = record.num_frames or self.frame_counter(video_path)
        frame_inds = self.sampler.sample(video_length)
        frames = self._load_frames(video_path, frame_inds)
        label = record.label

        if self.transform is not None:
            frames = self.transform(frames)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return frames, label

    def __len__(self):
        return len(self.record_set)

    @staticmethod
    def _load_frames(
        video_file: Path,
        frame_idx: Union[slice, List[slice], List[int]],
    ) -> Iterator[Image]:
        from torchvideo.internal.readers import default_loader

        return default_loader(video_file, frame_idx)


class ImageFolderVideoDataset(VideoDataset):
    """Dataset stored as a folder containing folders of images, where each folder
    represents a video.

    The folder hierarchy should look something like this: ::

        root/video1/frame_000001.jpg
        root/video1/frame_000002.jpg
        root/video1/frame_000003.jpg
        ...

        root/video2/frame_000001.jpg
        root/video2/frame_000002.jpg
        root/video2/frame_000003.jpg
        root/video2/frame_000004.jpg
        ...

    """

    def __init__(
        self,
        root_path: Union[str, Path],
        filename_template: str,
        filter: Optional[Callable[[Path], bool]] = None,
        label_set: Optional[LabelSet] = None,
        sampler: FrameSampler = _default_sampler(),
        transform: Optional[PILVideoTransform] = None,
        frame_counter: Optional[Callable[[Path], int]] = None,
    ):
        """

        Args:
            root_path: Path to dataset on disk. Contents of this folder should be
                example folders, each with frames named according to the
                ``filename_template`` argument.
            filename_template: Python 3 style formatting string describing frame
                filenames: e.g. ``"frame_{:06d}.jpg"`` for the example dataset in the
                class docstring.
            filter: Optional filter callable that decides whether a given example folder
                is to be included in the dataset or not.
            label_set: Optional label set for labelling examples.
            sampler: Optional sampler for drawing frames from each video.
            transform: Optional transform performed over the loaded clip.
            frame_counter: Optional callable used to determine the number of frames
                each video contains. The callable will be passed the path to a video
                folder and should return a positive integer representing the number of
                frames. This tends to be useful if you've precomputed the number of
                frames in a dataset.
        """
        super().__init__(root_path, label_set, sampler=sampler, transform=transform)
        self.video_dirs = sorted(
            [d for d in self.root_path.iterdir() if filter is None or filter(d)]
        )
        self.labels = self._label_examples(self.video_dirs, label_set)
        self.video_lengths = self._measure_video_lengths(self.video_dirs, frame_counter)
        self.filename_template = filename_template

    def __len__(self) -> int:
        return len(self.video_dirs)

    def __getitem__(
        self, index: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Label]]:
        video_folder = self.video_dirs[index]
        video_length = self.video_lengths[index]
        video_name = video_folder.name
        frames_idx = self.sampler.sample(video_length)
        frames = self._load_frames(frames_idx, video_folder)
        if self.transform is None:
            frames_tensor = PILVideoToTensor()(frames)
        else:
            frames_tensor = self.transform(frames)

        if self.labels is not None:
            label = self.labels[index]
            return frames_tensor, label
        else:
            return frames_tensor

    @staticmethod
    def _measure_video_lengths(
        video_dirs, frame_counter: Optional[Callable[[Path], int]]
    ):
        if frame_counter is None:
            return [len(list(video_dir.iterdir())) for video_dir in video_dirs]
        else:
            return [frame_counter(video_dir) for video_dir in video_dirs]

    @staticmethod
    def _label_examples(video_dirs, label_set: Optional[LabelSet]):
        if label_set is not None:
            return [label_set[video_dir.name] for video_dir in video_dirs]
        else:
            return None

    def _load_frames(
        self, frames_idx: Union[slice, List[slice], List[int]], video_folder: Path
    ) -> Iterator[Image]:
        frame_numbers = frame_idx_to_list(frames_idx)
        filepaths = [
            video_folder / self.filename_template.format(index + 1)
            for index in frame_numbers
        ]
        frames = (self._load_image(path) for path in filepaths)
        # shape: (n_frames, height, width, channels)
        return frames

    def _load_image(self, path: Path) -> Image:
        if not path.exists():
            raise ValueError("Image path {} does not exist".format(path))
        return PIL.Image.open(str(path))


class VideoFolderDataset(VideoDataset):
    """Dataset stored as a folder of videos, where each video is a single example
    in the dataset.

    The folder hierarchy should look something like this: ::

        root/video1.mp4
        root/video2.mp4
        ...
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        filter: Optional[Callable[[Path], bool]] = None,
        label_set: Optional[LabelSet] = None,
        sampler: FrameSampler = _default_sampler(),
        transform: Optional[PILVideoTransform] = None,
        frame_counter: Optional[Callable[[Path], int]] = None,
    ) -> None:
        """
        Args:
            root_path: Path to dataset folder on disk. The contents of this folder
                should be video files.
            filter: Optional filter callable that decides whether a given example video
                is to be included in the dataset or not.
            label_set: Optional label set for labelling examples.
            sampler: Optional sampler for drawing frames from each video.
            transform: Optional transform over the list of frames.
            frame_counter: Optional callable used to determine the number of frames
                each video contains. The callable will be passed the path to a video and
                should return a positive integer representing the number of frames.
                This tends to be useful if you've precomputed the number of frames in a
                dataset.
        """
        super().__init__(
            root_path, label_set=label_set, sampler=sampler, transform=transform
        )
        self.video_paths = self._get_video_paths(self.root_path, filter)
        self.labels = self._label_examples(self.video_paths, label_set)
        self.video_lengths = self._measure_video_lengths(
            self.video_paths, frame_counter
        )

    # TODO: This is very similar to ImageFolderVideoDataset consider merging into
    #  VideoDataset
    def __getitem__(
        self, index: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Label]]:
        video_file = self.video_paths[index]
        video_name = video_file.stem
        video_length = self.video_lengths[index]
        frames_idx = self.sampler.sample(video_length)
        frames = self._load_frames(frames_idx, video_file)
        if self.transform is not None:
            frames = self.transform(frames)

        if self.labels is not None:
            return frames, self.labels[index]
        else:
            return frames

    def __len__(self):
        return len(self.video_paths)

    @staticmethod
    def _measure_video_lengths(video_paths, frame_counter):
        if frame_counter is None:
            frame_counter = _get_videofile_frame_count
        return [frame_counter(vid_path) for vid_path in video_paths]

    @staticmethod
    def _label_examples(video_paths, label_set: Optional[LabelSet]):
        if label_set is None:
            return None
        else:
            return [label_set[video_path.name] for video_path in video_paths]

    @staticmethod
    def _get_video_paths(root_path, filter):
        return sorted(
            [
                root_path / child
                for child in root_path.iterdir()
                if _is_video_file(child) and (filter is None or filter(child))
            ]
        )

    @staticmethod
    def _load_frames(
        frame_idx: Union[slice, List[slice], List[int]], video_file: Path
    ) -> Iterator[Image]:
        from torchvideo.internal.readers import default_loader

        return default_loader(video_file, frame_idx)


class GulpVideoDataset(VideoDataset):
    """GulpIO Video dataset.

    The folder hierarchy should look something like this: ::

        root/data_0.gulp
        root/data_1.gulp
        ...

        root/meta_0.gulp
        root/meta_1.gulp
        ...
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        filter: Optional[Callable[[str], bool]] = None,
        label_field: Optional[str] = None,
        label_set: Optional[LabelSet] = None,
        sampler: FrameSampler = _default_sampler(),
        transform: Optional[NDArrayVideoTransform] = None,
    ):
        """
        Args:
            root_path: Path to GulpIO dataset folder on disk. The ``.gulp`` and
                ``.gmeta`` files are direct children of this directory.
            filter: Filter function that determines whether a video is included into
                the dataset. The filter is called on each video id, and should return
                ``True`` to include the video, and ``False`` to exclude it.
            label_field: Meta data field name that stores the label of an example,
                this is used to construct a :class:`GulpLabelSet` that performs the
                example labelling. Defaults to ``'label'``.
            label_set: Optional label set for labelling examples. This is mutually
                exclusive with ``label_field``.
            sampler: Optional sampler for drawing frames from each video.
            transform: Optional transform over the :class:`ndarray` with layout
                ``THWC``. Note you'll probably want to remap the channels to ``CTHW`` at
                the end of this transform.
        """
        from gulpio import GulpDirectory

        self.gulp_dir = GulpDirectory(str(root_path))
        label_set = self._get_label_set(self.gulp_dir, label_field, label_set)
        super().__init__(
            root_path, label_set=label_set, sampler=sampler, transform=transform
        )
        self._video_ids = self._get_video_ids(self.gulp_dir, filter)
        self.labels = self._label_examples(self._video_ids, self.label_set)

    @staticmethod
    def _label_examples(video_ids: List[str], label_set: Optional[LabelSet]):
        if label_set is None:
            return None
        else:
            return [label_set[video_id] for video_id in video_ids]

    def __len__(self):
        return len(self._video_ids)

    def __getitem__(self, index) -> Union[torch.Tensor, Tuple[torch.Tensor, Label]]:
        id_ = self._video_ids[index]
        frame_count = self._get_frame_count(id_)
        frame_idx = self.sampler.sample(frame_count)
        if isinstance(frame_idx, slice):
            frames = self._load_frames(id_, frame_idx)
        elif isinstance(frame_idx, list):
            if isinstance(frame_idx[0], slice):
                frames = np.concatenate(
                    [self._load_frames(id_, slice_) for slice_ in frame_idx]
                )
            elif isinstance(frame_idx[0], numbers.Number):
                frames = np.concatenate(
                    [
                        self._load_frames(id_, slice(index, index + 1))
                        for index in frame_idx
                    ]
                )
            else:
                raise TypeError(
                    "frame_idx was a list of {} but we only support "
                    "int and slice elements".format(type(frame_idx[0]).__name__)
                )
        else:
            raise TypeError(
                "frame_idx was of type {} but we only support slice, "
                "List[slice], List[int]".format(type(frame_idx).__name__)
            )

        if self.transform is not None:
            frames = self.transform(frames)
        else:
            frames = torch.Tensor(np.rollaxis(frames, -1, 0)).div_(255)

        if self.labels is not None:
            label = self.labels[index]
            return frames, label
        else:
            return frames

    @staticmethod
    def _get_video_ids(gulp_dir, filter_fn: Callable[[str], bool]) -> List[str]:
        return sorted(
            [
                id_
                for id_ in gulp_dir.merged_meta_dict.keys()
                if filter_fn is None or filter_fn(id_)
            ]
        )

    @staticmethod
    def _get_label_set(gulp_dir, label_field: str, label_set: LabelSet):
        if label_field is None:
            label_field = "label"
        if label_set is None:
            label_set = GulpLabelSet(gulp_dir.merged_meta_dict, label_field=label_field)
        return label_set

    def _load_frames(self, id_: str, frame_idx: slice) -> np.ndarray:
        frames, _ = self.gulp_dir[id_, frame_idx]
        return np.array(frames, dtype=np.uint8)

    def _get_frame_count(self, id_: str):
        info = self.gulp_dir.merged_meta_dict[id_]
        return len(info["frame_info"])


class StaticFrameCounter:
    def __init__(self, num_frames=None):
        self.num_frames = num_frames

    def __call__(self, input):
        if self.num_frames is None:
            self.num_frames = _get_videofile_frame_count(input)
        return self.num_frames
