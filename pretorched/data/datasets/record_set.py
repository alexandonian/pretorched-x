
import json
import functools
from abc import ABC
from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, Optional

Label = Any
# VideoRecord = namedtuple('VideoRecord', ['path', 'label'])
# MultiLabelVideoRecord = namedtuple('MultiLabelVideoRecord', ['path', 'labels'])
FrameFolderRecord = namedtuple('FrameFolderRecord', ['path', 'num_frames', 'label'])


class VideoRecord:
    """Represents a video record.
    A video record has the following properties:
        path (str): path to directory containing frames.
        num_frames (int): number of frames in path dir.
        label (int): primary label associated with video.
    """

    def __init__(self, data):
        self.data = data

    @property
    def path(self):
        return self.data['path']

    @property
    def filename(self):
        return self.data['filename']

    @property
    def num_frames(self):
        return self.data.get('num_frames')

    @property
    def height(self):
        return self.data.get('height')

    @property
    def width(self):
        return self.data.get('width')

    @property
    def size(self):
        return (self.height, self.width)

    @property
    def label(self):
        return int(self.data['label'])

    @property
    def category(self):
        return self.data['category']

    def todict(self):
        return self.data

    def __hash__(self):
        return hash(self.data.values())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.path == other.path
        else:
            return False


class MultiLabelVideoRecord(VideoRecord):

    @property
    def labels(self):
        return self.data['labels']

    @property
    def categories(self):
        return self.data['categories']

    @property
    def label(self):
        return self.labels[0]

    @property
    def category(self):
        return self.categories[0]


class RecordSet:

    def __init__(self, metafile, record_func=VideoRecord, blacklist_file=None):
        self.records = []
        self.metafile = metafile
        self.record_func = record_func
        with open(metafile) as f:
            self.data = json.load(f)

        self.blacklist = []
        if blacklist_file is not None:
            with open(blacklist_file) as f:
                self.blacklist.extend(json.load(f))

        self.records = []
        self.records_dict = defaultdict(list)
        self.blacklist_records = []
        for rec_data in self.data:
            record = self.record_func(rec_data)
            if record.path not in self.blacklist:
                self.records.append(record)
                self.records_dict[record.label].append(record)
            else:
                self.blacklist_records.append(record)

    def __getitem__(self, idx: int) -> VideoRecord:
        return self.records[idx]

    def __len__(self):
        return len(self.records)


MultiLabelRecordSet = functools.partial(RecordSet, record_func=MultiLabelVideoRecord)

# class RecordSet:

#     def __init__(self, metafile, sep: Optional[str] = ' '):
#         self.records = []
#         self.metafile = metafile
#         with open(metafile) as f:
#             for line in f:
#                 path, label = line.strip().split(sep)
#                 self.records.append(VideoRecord(path, int(label)))

#     def __getitem__(self, idx: int) -> VideoRecord:
#         return self.records[idx]

#     def __len__(self):
#         return len(self.records)


class _MultiLabelRecordSet:

    def __init__(self, metafile,
                 category_filename,
                 sep: Optional[str] = ','):
        self.records = []
        self.metafile = metafile
        self.category_filename = category_filename

        with open(category_filename) as f:
            self.cat2label = {k: int(v) for k, v in dict(line.strip().split(',') for line in f).items()}

        with open(metafile) as f:
            for line in f:
                path, *categories = line.strip().split(sep)
                self.records.append(MultiLabelVideoRecord(path, [int(self.cat2label[cat]) for cat in categories]))

    def __getitem__(self, idx: int) -> MultiLabelVideoRecord:
        return self.records[idx]

    def __len__(self):
        return len(self.records)


class FrameRecordSet(RecordSet):

    def __init__(self, metafile, sep: Optional[str] = ' '):
        self.records = []
        self.metafile = metafile
        with open(metafile) as f:
            for line in f:
                path, num_frames, label = line.strip().split(sep)
                self.records.append(FrameFolderRecord(path + '.mp4', int(num_frames), int(label)))

    def __getitem__(self, idx: int) -> FrameFolderRecord:
        return self.records[idx]


class LabelSet(ABC):  # pragma: no cover
    """Abstract base class that all ``LabelSets`` inherit from

    If you are implementing your own ``LabelSet``, you should inherit from this
    class."""

    def __getitem__(self, video_name: str) -> Label:
        """
        Args:
            video_name: The filename or id of the video

        Returns:
            The corresponding label
        """
        raise NotImplementedError()


class DummyLabelSet(LabelSet):
    """A dummy label set that returns the same label regardless of video"""

    def __init__(self, label: Label = 0):
        """
        Args:
            label: The label given to any video
        """
        self.label = label

    def __getitem__(self, video_name) -> Label:
        return self.label

    def __repr__(self):
        return self.__class__.__name__ + "(label={!r})".format(self.label)


class LambdaLabelSet(LabelSet):
    """A label set that wraps a function used to retrieve a label for an example"""

    def __init__(self, labeller_fn: Callable[[str], Label]):
        """
        Args:
            labeller_fn: Function for labelling examples.
        """
        self._labeller_fn = labeller_fn

    def __getitem__(self, video_name: str) -> Label:
        return self._labeller_fn(video_name)


class GulpLabelSet(LabelSet):
    """LabelSet for GulpIO datasets where the label is contained within the metadata of
    the gulp directory. Assuming you've written the label of each video to a field
    called ``'label'`` in the metadata you can create a LabelSet like:
    ``GulpLabelSet(gulp_dir.merged_meta_dict, label_field='label')``
    """

    def __init__(self, merged_meta_dict: Dict[str, Any], label_field: str = "label"):
        self.merged_meta_dict = merged_meta_dict
        self.label_field = label_field

    def __getitem__(self, video_name: str) -> Label:
        # The merged meta dict has the form: { video_id: { meta_data: [{ meta... }] }}
        video_meta_data = self.merged_meta_dict[video_name]["meta_data"][0]
        return video_meta_data[self.label_field]
