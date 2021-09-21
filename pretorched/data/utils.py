import functools
import itertools
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Process

# from multiprocessing.pool import ThreadPool as Pool
from multiprocessing.pool import Pool as Pool
from typing import List, Union

import ffmpeg
import numpy as np
import PIL
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

try:
    from moviepy.editor import ImageSequenceClip
    from moviepy.video.io.html_tools import ipython_available

    moviepy_available = True
except ImportError:
    moviepy_available = False


def _slice_to_list(slice_: slice) -> List[int]:
    step = 1 if slice_.step is None else slice_.step
    return list(range(slice_.start, slice_.stop, step))


def _is_int(maybe_int):
    try:
        return int(maybe_int) == maybe_int
    except TypeError:
        pass
    return False


def frame_idx_to_list(frames_idx: Union[slice, List[slice], List[int]]) -> List[int]:
    """Convert a frame_idx object to a list of indices. Useful for testing.

    Args:
        frames_idx: Frame indices represented as a slice, list of slices, or list of
            ints.

    Returns:
        frame idx as a list of ints.

    """
    if isinstance(frames_idx, list):
        if isinstance(frames_idx[0], slice):
            return list(
                itertools.chain.from_iterable([_slice_to_list(s) for s in frames_idx])
            )
        if _is_int(frames_idx[0]):
            return frames_idx
    if isinstance(frames_idx, slice):
        return _slice_to_list(frames_idx)
    raise ValueError(
        "Can't handle {} objects, must be slice, List[slice], or List[int]".format(
            type(frames_idx)
        )
    )


def compute_sample_length(clip_length, step_size):
    """Compute the number of frames to be sampled for a clip.

     Clip is of length ``clip_length`` with frame step size of ``step_size``
     to be generated.

    Args:
        clip_length: Number of frames to sample
        step_size: Number of frames to skip in between adjacent frames in the output

    Returns:
        Number of frames to sample to read a clip of length ``clip_length`` while
        skipping ``step_size - 1`` frames.

    """
    return 1 + step_size * (clip_length - 1)


def show_video(frames: Union[torch.Tensor, np.ndarray], fps=30):
    r"""Show video.

    Args:
        frames: Either a :class:`torch.Tensor` or :class:`numpy.ndarray` of shape
            :math:`T \times C \times H \times W` or a list of :class:`PIL.Image.Image`s
        fps (optional): Frame rate of video

    Returns:

    """
    if not moviepy_available:
        raise ModuleNotFoundError("moviepy not found, please install moviepy")
    # Input format: (C, T, H, W)
    # Desired shape: (T, H, W, C)
    if isinstance(frames, torch.Tensor):
        frames = torch.clamp((frames * 255), 0, 255).to(torch.uint8)
        frames_list = list(frames.permute(1, 2, 3, 0).cpu().numpy())
    elif isinstance(frames, np.ndarray):
        frames_list = list(np.roll(frames, -1))
    elif isinstance(frames, list):
        if not isinstance(frames[0], Image.Image):
            raise ValueError("Expected a list of PIL Images when passed a sequence")
        frames_list = list(map(np.array, frames))
    else:
        raise TypeError(
            "Unknown type: {}, expected np.ndarray, torch.Tensor, "
            "or sequence of PIL.Image.Image".format(type(frames).__name__)
        )

    video = ImageSequenceClip(frames_list, fps=fps)
    if ipython_available:
        return video.ipython_display()
    else:
        return video.show()


def extract_frames(video, video_root='', frame_root='', tmpl='{:06d}.jpg', fps=25):
    name, _ = os.path.splitext(os.path.basename(video))
    in_filename = os.path.join(video_root, video)
    out_filename = os.path.join(frame_root, name, tmpl)
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    (
        ffmpeg.input(in_filename)
        .filter('fps', fps=fps, round='up')
        .output(out_filename)
        .global_args('-loglevel', 'error')
        .run()
    )


def _extract_frames(video, video_root='', frame_root='', tmpl='%06d.jpg'):
    """Extract frames from video using call to ffmpeg."""
    print(f'extracting frames from {video}')
    return subprocess.run(
        [
            'ffmpeg',
            '-i',
            os.path.join(video_root, video),
            '-vf',
            'scale=320:-1,fps=25',
            os.path.join(frame_root, video.rstrip('.mp4'), tmpl),
        ]
    )


def _make_collage(frame_dir, collage_root='', file_tmpl='{:06d}.jpg'):
    """Loads extracted frames from dir and assembles them into a collage."""
    files = os.listdir(frame_dir)
    transform = torchvision.transforms.ToTensor()
    try:
        frames = torch.stack(
            [
                transform(PIL.Image.open(os.path.join(frame_dir, f)).convert('RGB'))
                for f in files
            ]
        )
    except RuntimeError as e:
        print(e, f'had trouble loading frames for video: {frame_dir}')
        return
    name = '/'.join(frame_dir.split('/')[-2:])
    print(f'creating collage: {name}')
    collage_filename = os.path.join(collage_root, name + '.jpg')
    torchvision.utils.save_image(frames, collage_filename)


def _extraction_closure(video_root, frame_root, collage_root):
    """Closure that returns function to extract frames for video list."""

    def func(video_list):
        for video in video_list:
            frame_dir = video.rstrip('.mp4')
            frame_path = os.path.join(frame_root, frame_dir)
            os.makedirs(frame_path, exist_ok=True)
            extract_frames(video, video_root, frame_root)
            make_collage(frame_path, collage_root)

    return func


def extraction_closure(video_root, frame_root):
    """Closure that returns function to extract frames for video list."""

    def func(video_list):
        for video in video_list:
            frame_dir = video.rstrip('.mp4')
            frame_path = os.path.join(frame_root, frame_dir)
            os.makedirs(frame_path, exist_ok=True)
            extract_frames(video, video_root, frame_root)

    return func


def videos_to_frames(video_root, frame_root, num_threads=100):
    """videos_to_frames."""
    categories = os.listdir(video_root)
    videos = [
        [os.path.join(cat, v) for v in os.listdir(os.path.join(video_root, cat))]
        for cat in categories
    ]

    # func = extraction_closure(video_root, frame_root)
    func = functools.partial(
        extract_frames, video_root=video_root, frame_root=frame_root
    )
    pool = Pool(num_threads)
    pool.map(func, videos)


def frames_to_collages(frame_root, collage_root, num_threads=100):
    categories = os.listdir(frame_root)
    for cat in categories:
        os.makedirs(os.path.join(collage_root, cat), exist_ok=True)
    # frame_dirs = [
    # [os.path.join(cat, f)
    #  for f in os.listdir(os.path.join(frame_root, cat))
    #  if not os.path.exists(os.path.join(collage_root, cat, f'{f}.jpg'))]
    # for cat in categories]
    frame_dirs = [
        [os.path.join(cat, f) for f in os.listdir(os.path.join(frame_root, cat))]
        for cat in categories
    ]

    func = collages_closure(frame_root, collage_root)
    pool = Pool(num_threads)
    pool.map(func, frame_dirs)


def get_info(filename, strict=False):
    try:
        probe = ffmpeg.probe(filename)
    except ffmpeg.Error as e:
        print(e.stderr, file=sys.stderr)
        if strict:
            sys.exit(1)
        return {}

    video_stream = next(
        (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None
    )
    if video_stream is None:
        print('No video stream found', file=sys.stderr)
        if strict:
            sys.exit(1)
        return {}

    return {
        'width': int(video_stream['width']),
        'height': int(video_stream['height']),
        'num_frames': int(video_stream['nb_frames']),
        'fps': eval(video_stream['r_frame_rate']),
    }


def get_audio_info(filename, strict=False):
    try:
        probe = ffmpeg.probe(filename)
    except ffmpeg.Error as e:
        print(e.stderr, file=sys.stderr)
        if strict:
            sys.exit(1)
        return {}
    audio_stream = next(
        (stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None
    )
    # print(audio_stream)
    if audio_stream is None:
        print('No audio stream found', file=sys.stderr)
        if strict:
            sys.exit(1)
        return {}

    return {
        'sample_rate': int(audio_stream['sample_rate']),
        'codec_name': audio_stream['codec_name'],
        'channels': int(audio_stream['channels']),
        'duration': float(audio_stream['duration']),
    }


def get_size(filename):
    info = get_info(filename)
    return info['height'], info['width']


def load_video(filename, fps=25, height=None, width=None):
    if any(x is None for x in (height, width)):
        height, width = get_size(filename)

    out, _ = (
        ffmpeg.input(filename)
        .filter('fps', fps=fps, round='up')
        .filter('scale', width, height)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True, capture_stderr=True)
    )
    return np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])


def make_grid(tensor, nrow=8):
    """Make a grid of images."""
    tensor = np.array(tensor)
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1]), int(tensor.shape[2])
    grid = np.zeros((height * ymaps, width * xmaps, 3), dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[y * height : (y + 1) * height, x * width : (x + 1) * width] = tensor[k]
            k += 1
    return grid


def make_collage(video_filename, collage_filename=None, smallest_side_size=320):
    if collage_filename is None:
        collage_filename = video_filename.rstrip('.mp4') + '.jpg'
    size = get_size(video_filename)
    scale = smallest_side_size / min(*size)
    h, w = map(int, [s * scale for s in size])
    video = load_video(video_filename, height=h, width=w)
    montage = make_grid(video)
    im = Image.fromarray(montage)
    im.save(collage_filename)
    print(f'Saving: {collage_filename}')


def _videos_to_collages(args):
    video, video_root, collage_root = args
    video_path = os.path.join(video_root, video)
    collage_name = video.rstrip('.mp4') + '.jpg'
    collage_path = os.path.join(collage_root, collage_name)
    make_collage(video_path, collage_filename=collage_path)


def videos_to_collages(video_root, collage_root, num_workers=100):
    categories = os.listdir(video_root)
    for cat in categories:
        os.makedirs(os.path.join(collage_root, cat), exist_ok=True)
    videos = [
        (os.path.join(cat, f), video_root, collage_root)
        for cat in categories
        for f in os.listdir(os.path.join(video_root, cat))
        if not os.path.exists(
            os.path.join(collage_root, cat, f.replace('.mp4', '.jpg'))
        )
    ]

    random.shuffle(videos)
    pool = Pool(num_workers)
    pool.map(_videos_to_collages, videos)


def frames_to_video(input, output, pattern_type='glob', framerate=30, vcodec='libx264'):
    (
        ffmpeg.input(input, pattern_type=pattern_type, framerate=framerate)
        .output(output, vcodec=vcodec)
        .global_args('-loglevel', 'error', '-n')
        .run()
    )


def array_to_video(images, filename, framerate=30, vcodec='libx264'):
    """Save numpy array to a video file.

    Args:
        images: [n, h, w, c] uint8 in range [0..255].
    """
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    n, height, width, channels = images.shape
    print(n, height, width, channels)
    process = (
        ffmpeg.input(
            'pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height)
        )
        .output(filename, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
        .global_args('-loglevel', 'error')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(frame.astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()


def async_array_to_video(images, filename):
    p = Process(
        target=array_to_video,
        args=(images, filename),
        kwargs={'vcodec': 'libx264'},
    )
    p.start()


def downsample_video(input, output, smallest_side_size=320, vcodec='libx264') -> None:
    size = get_size(input)
    scale = smallest_side_size / min(*size)
    h, w = map(int, [((s * scale) // 2) * 2 for s in size])
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    (
        ffmpeg.input(input)
        .filter('scale', w, h)
        .output(output, vcodec=vcodec)
        .global_args('-loglevel', 'error', '-n')
        .run()
    )


def encode_video(filename, outname, vcodec='libx264', crf=None, scale=1.0, fps=None):
    outkwargs = {}
    if crf is not None:
        outkwargs['crf'] = crf
    if vcodec is not None:
        outkwargs['vcodec'] = vcodec
    stream = ffmpeg.input(filename)
    if (scale is not None) and (scale != 1.0):
        stream = ffmpeg.filter(stream, 'scale', f'{scale}*iw', f'{scale}*ih')
    if fps is not None:
        stream = ffmpeg.filter(stream, 'fps', fps=fps, round='up')
    stream = ffmpeg.output(stream, outname, **outkwargs).global_args(
        '-loglevel', 'error', '-n'
    )
    ffmpeg.run(stream)


def reencode_video(filename, vcodec='libx264', crf=18, scale=1.0, fps=None):
    with tempfile.TemporaryDirectory() as tmpdirname:
        outname = os.path.join(tmpdirname, os.path.basename(filename))
        encode_video(filename, outname, vcodec=vcodec, crf=crf, scale=scale, fps=fps)
        shutil.copy(outname, filename)


def reencode_videos(*filenames, num_workers=12):
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(reencode_video, filenames), total=len(filenames)))
        pool.close()
        pool.join()
