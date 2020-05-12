import pytest
import torch

from pretorched.runners import core, config as cfg


DATA_ROOT = cfg.DATA_ROOT
MAX_ITERS = 10
BATCH_SIZE = 32

SKIP_VIDEO = False
SKIP_VIDEO_DATASET = False
SKIP_VIDEO_ZIP_DATASET = False
SKIP_DATASET = False
SKIP_DATALOADER = True


@pytest.mark.parametrize('name, split, size, dataset_type', [
    ('ImageNet', 'train', 224, 'ImageFolder'),
    ('ImageNet', 'val', 224, 'ImageFolder'),
    ('Places365', 'train', 224, 'ImageFolder'),
    ('Places365', 'val', 224, 'ImageFolder'),
])
def test_get_dataset(name, split, size, dataset_type):
    root = cfg.get_root_dirs(name, dataset_type=dataset_type,
                             data_root=DATA_ROOT)

    dataset = core.get_dataset(name=name, root=root,
                               split=split, size=224,
                               dataset_type=dataset_type)
    img, label = dataset[0]
    assert label < cfg.NUM_CLASSES[name]
    assert img.shape == torch.Size((3, size, size))


@pytest.mark.parametrize('name, split, size, dataset_type', [
    ('Hybrid1365', 'train', 224, 'ImageFolder'),
    ('Hybrid1365', 'val', 224, 'ImageFolder'),
])
def test_get_hybrid_dataset(name, split, size, dataset_type):
    root = cfg.get_root_dirs(name, dataset_type=dataset_type,
                             data_root=DATA_ROOT)

    dataset = core.get_hybrid_dataset(name=name, root=root,
                                      split=split, size=224,
                                      dataset_type=dataset_type)
    img, label = dataset[0]
    assert label < cfg.NUM_CLASSES[name]
    assert img.shape == torch.Size((3, size, size))


@pytest.mark.skipif(SKIP_DATALOADER, reason='Not needed.')
@pytest.mark.parametrize('name, split, size, dataset_type', [
    ('ImageNet', 'train', 224, 'ImageFolder'),
    ('ImageNet', 'val', 224, 'ImageFolder'),
    ('Places365', 'train', 224, 'ImageFolder'),
    ('Places365', 'val', 224, 'ImageFolder'),
    ('Hybrid1365', 'train', 224, 'ImageFolder'),
    ('Hybrid1365', 'val', 224, 'ImageFolder'),
])
def test_get_dataloader(name, split, size, dataset_type):

    loader = core.get_dataloader(
        name, data_root=DATA_ROOT, split=split, size=size, dataset_type=dataset_type,
        batch_size=BATCH_SIZE, num_workers=4, shuffle=True, load_in_mem=False, pin_memory=True,
        drop_last=True, distributed=False)

    for i, (x, y) in enumerate(loader):
        if i >= MAX_ITERS:
            break

        assert y.shape == torch.Size((BATCH_SIZE,))
        assert x.shape == torch.Size((BATCH_SIZE, 3, size, size))


@pytest.mark.skipif(SKIP_VIDEO_DATASET, reason='Not needed')
@pytest.mark.parametrize('name, split, segment_count, size, dataset_type, record_set_type', [
    ('Moments', 'train', 16, 224, 'VideoRecordDataset', 'RecordSet'),
    ('Moments', 'val', 16, 224, 'VideoRecordDataset', 'RecordSet'),
    ('Moments', 'test', 16, 224, 'VideoRecordDataset', 'RecordSet'),
    ('MultiMoments', 'train', 16, 224, 'VideoRecordDataset', 'MultiLabelRecordSet'),
    ('MultiMoments', 'val', 16, 224, 'VideoRecordDataset', 'MultiLabelRecordSet'),
    ('MultiMoments', 'test', 16, 224, 'VideoRecordDataset', 'MultiLabelRecordSet'),
    ('Kinetics', 'train', 16, 224, 'VideoRecordDataset', 'RecordSet'),
    ('Kinetics', 'val', 16, 224, 'VideoRecordDataset', 'RecordSet'),
    ('Kinetics', 'test', 16, 224, 'VideoRecordDataset', 'RecordSet'),
])
def test_get_video_dataset(name, split, segment_count, size, dataset_type, record_set_type):

    dataset = core.get_video_dataset(
        name, data_root=DATA_ROOT, split=split, size=size, dataset_type=dataset_type,
        record_set_type=record_set_type, load_in_mem=False, segment_count=segment_count)

    for i, (x, y) in enumerate(dataset):
        if i >= MAX_ITERS:
            break

        assert y < cfg.NUM_CLASSES[name]
        assert x.shape == torch.Size((3, segment_count, size, size))


@pytest.mark.skipif(SKIP_VIDEO_ZIP_DATASET, reason='Not needed')
@pytest.mark.parametrize('name, split, segment_count, size, dataset_type, record_set_type', [
    ('Moments', 'train', 16, 224, 'VideoRecordZipDataset', 'RecordSet'),
    ('Moments', 'val', 16, 224, 'VideoRecordZipDataset', 'RecordSet'),
    ('Moments', 'test', 16, 224, 'VideoRecordZipDataset', 'RecordSet'),
    ('MultiMoments', 'train', 16, 224, 'VideoRecordZipDataset', 'MultiLabelRecordSet'),
    ('MultiMoments', 'val', 16, 224, 'VideoRecordZipDataset', 'MultiLabelRecordSet'),
    ('Kinetics', 'train', 16, 224, 'VideoRecordZipDataset', 'RecordSet'),
    ('Kinetics', 'val', 16, 224, 'VideoRecordZipDataset', 'RecordSet'),
    ('Kinetics', 'test', 16, 224, 'VideoRecordZipDataset', 'RecordSet'),
])
def test_get_video_zip_dataset(name, split, segment_count, size, dataset_type, record_set_type):

    dataset = core.get_video_dataset(
        name, data_root=DATA_ROOT, split=split, size=size, dataset_type=dataset_type,
        record_set_type=record_set_type, load_in_mem=False, segment_count=segment_count)

    for i, (x, y) in enumerate(dataset):
        if i >= MAX_ITERS:
            break

        assert y < cfg.NUM_CLASSES[name]
        assert x.shape == torch.Size((3, segment_count, size, size))


@pytest.mark.skipif(SKIP_DATALOADER, reason='Not needed.')
@pytest.mark.parametrize('name, split, segment_count, size, dataset_type, record_set_type', [
    ('Moments', 'train', 16, 224, 'VideoRecordDataset', 'RecordSet'),
    ('Moments', 'val', 16, 224, 'VideoRecordDataset', 'RecordSet'),
    ('Moments', 'test', 16, 224, 'VideoRecordDataset', 'RecordSet'),
    ('MultiMoments', 'train', 16, 224, 'VideoRecordDataset', 'MultiLabelRecordSet'),
    ('MultiMoments', 'val', 16, 224, 'VideoRecordDataset', 'MultiLabelRecordSet'),
    ('MultiMoments', 'test', 16, 224, 'VideoRecordDataset', 'MultiLabelRecordSet'),
    ('Kinetics', 'train', 16, 224, 'VideoRecordDataset', 'RecordSet'),
    ('Kinetics', 'val', 16, 224, 'VideoRecordDataset', 'RecordSet'),
    ('Kinetics', 'test', 16, 224, 'VideoRecordDataset', 'RecordSet'),
])
def test_get_video_dataloader(name, split, segment_count, size, dataset_type, record_set_type):

    loader = core.get_video_dataloader(
        name, data_root=DATA_ROOT, split=split, size=size, dataset_type=dataset_type,
        record_set_type=record_set_type, batch_size=BATCH_SIZE, num_workers=8, shuffle=True,
        load_in_mem=False, pin_memory=True, drop_last=False, distributed=False, segment_count=segment_count)

    for i, (x, y) in enumerate(loader):
        if i >= MAX_ITERS:
            break

        assert y.shape == torch.Size((BATCH_SIZE,))
        assert x.shape == torch.Size((BATCH_SIZE, 3, segment_count, size, size))

@pytest.mark.skipif(SKIP_DATALOADER, reason='Not needed.')
@pytest.mark.parametrize('name, split, segment_count, size, dataset_type, record_set_type', [
    ('Moments', 'train', 16, 224, 'VideoRecordZipDataset', 'RecordSet'),
    ('Moments', 'val', 16, 224, 'VideoRecordZipDataset', 'RecordSet'),
    ('Moments', 'test', 16, 224, 'VideoRecordZipDataset', 'RecordSet'),
    ('MultiMoments', 'train', 16, 224, 'VideoRecordZipDataset', 'MultiLabelRecordSet'),
    ('MultiMoments', 'val', 16, 224, 'VideoRecordZipDataset', 'MultiLabelRecordSet'),
    ('MultiMoments', 'test', 16, 224, 'VideoRecordZipDataset', 'MultiLabelRecordSet'),
    ('Kinetics', 'train', 16, 224, 'VideoRecordZipDataset', 'RecordSet'),
    ('Kinetics', 'val', 16, 224, 'VideoRecordZipDataset', 'RecordSet'),
    ('Kinetics', 'test', 16, 224, 'VideoRecordZipDataset', 'RecordSet'),
])
def test_get_video_zip_dataloader(name, split, segment_count, size, dataset_type, record_set_type):

    loader = core.get_video_dataloader(
        name, data_root=DATA_ROOT, split=split, size=size, dataset_type=dataset_type,
        record_set_type=record_set_type, batch_size=BATCH_SIZE, num_workers=8, shuffle=True,
        load_in_mem=False, pin_memory=True, drop_last=False, distributed=False, segment_count=segment_count)

    for i, (x, y) in enumerate(loader):
        if i >= MAX_ITERS:
            break

        assert y.shape == torch.Size((BATCH_SIZE,))
        assert x.shape == torch.Size((BATCH_SIZE, 3, segment_count, size, size))