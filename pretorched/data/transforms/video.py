import numbers
import numpy as np

import PIL
import random
from typing import List, Tuple, Union, Sequence, Callable, Iterator, Iterable, Optional

import torch
from PIL.Image import Image
import torchvision.transforms.functional as F
import torchvision.transforms.transforms as T
from . import functional as VF
from collections import namedtuple


ImageShape = namedtuple("ImageSize", ["height", "width"])


class RandomCropVideo:
    """Crop the given Video (composed of PIL Images) at a random location.

    Args:
        size: Desired output size of the crop. If ``size`` is an
            int instead of sequence like ``(h, w)``, a square crop ``(size, size)`` is
            made.
        padding: Optional padding on each border
            of the image. Default is ``None``, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed: Whether to pad the image if smaller than the
            desired size to avoid raising an exception.
        fill: Pixel fill value for constant fill. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the ``padding_mode`` is ``'constant'``.
        padding_mode: Type of padding. Should be one of: ``'constant'``, ``'edge'``,
            ``'reflect'`` or ``'symmetric'``.

             - ``'constant'``: pads with a constant value, this value is specified with
               fill
             - ``'edge'``: pads with the last value on the edge of the image
             - ``'reflect'``: pads with reflection of image (without repeating the last
               value on the edge) padding ``[1, 2, 3, 4]`` with 2 elements on both sides
                in reflect mode will result in ``[3, 2, 1, 2, 3, 4, 3, 2]``.
             - ``'symmetric'``: pads with reflection of image (repeating the last value
               on the edge) padding ``[1, 2, 3, 4]`` with 2 elements on both sides in
               symmetric mode will result in ``[2, 1, 1, 2, 3, 4, 4, 3]``.

    """

    def __init__(
        self,
        size: Union[Tuple[int, int], int],
        padding: Optional[Union[Tuple[int, int, int, int], Tuple[int, int]]] = None,
        pad_if_needed: bool = False,
        fill: int = 0,
        padding_mode: str = "constant",
    ):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(
        self, frames: Union[Iterator[Image], Iterable[Image]]
    ) -> Iterator[Image]:
        try:
            frames = frames.__iter__()
        except AttributeError:
            pass

        frame = self._maybe_pad(next(frames))
        i, j, h, w = T.RandomCrop.get_params(frame, self.size)

        yield F.crop(frame, i, j, h, w)
        for frame in frames:
            yield self._transform_img(frame, i, j, h, w)

    def _transform_img(self, frame: Image, i: int, j: int, h: int, w: int) -> Image:
        return F.crop(self._maybe_pad(frame), i, j, h, w)

    def _maybe_pad(self, frame: Image):
        if self.padding is not None:
            frame = F.pad(frame, self.padding, self.fill, self.padding_mode)
        # pad the width if needed
        frame_width = frame.size[0]
        desired_width = self.size[1]
        if self.pad_if_needed and frame_width < desired_width:
            horizontal_padding = desired_width - frame_width
            frame = F.pad(frame, (horizontal_padding, 0), self.fill, self.padding_mode)
        # pad the height if needed
        frame_height = frame.size[1]
        desired_height = self.size[0]
        if self.pad_if_needed and frame_height < desired_height:
            vertical_padding = desired_height - frame_height
            frame = F.pad(frame, (0, vertical_padding), self.fill, self.padding_mode)
        return frame

    def __repr__(self) -> str:
        return (
            self.__class__.__name__ + "(size={size!r}, padding={padding!r}, "
            "pad_if_needed={pad_if_needed!r}, "
            "fill={fill!r}, padding_mode={padding_mode!r})".format(
                size=tuple(self.size),
                padding=self.padding,
                pad_if_needed=self.pad_if_needed,
                fill=self.fill,
                padding_mode=self.padding_mode,
            )
        )


class CenterCropVideo:
    """Crops the given video (composed of PIL Images) at the center of the frame.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            ``int`` instead of sequence like ``(h, w)``, a square crop ``(size, size)``
            is made.
    """

    def __init__(self, size: Union[Tuple[int, int], int]):
        self._image_transform = T.CenterCrop(size)

    def __call__(self, frames: Iterator[Image]) -> Iterator[Image]:
        for frame in frames:
            yield self._image_transform(frame)

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(size={0})".format(self._image_transform.size)


class RandomHorizontalFlipVideo:
    """Horizontally flip the given video (composed of PIL Images) randomly with a given
    probability :math:`p`.

    Args:
        p (float): probability of the image being flipped.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(
        self, frames: Union[Iterator[Image], Iterable[Image]]
    ) -> Iterator[Image]:
        if random.random() < self.p:
            flip = True
        else:
            flip = False
        for frame in frames:
            if flip:
                yield F.hflip(frame)
            else:
                yield frame

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(p={})".format(self.p)


class NormalizeVideo:
    r"""

    Normalise ``torch.*Tensor`` :math:`t` given mean:
    :math:`M = (\mu_1, \ldots, \mu_n)`
    and std:
    :math:`\Sigma = (\sigma_1, \ldots, \sigma_n)`:
    :math:`t'_c = \frac{t_c - M_c}{\Sigma_c}`

    Args:
        mean: Sequence of means for each channel, or a single mean applying to all
            channels.
        std: Sequence of standard deviations for each channel, or a single standard
            deviation applying to all channels.
        inplace: Whether or not to perform the operation in place without allocating
            a new tensor.
    """

    def __init__(
        self,
        mean: Union[Sequence[numbers.Number], numbers.Number],
        std: Union[Sequence[numbers.Number], numbers.Number],
        inplace: bool = False,
    ):
        self.mean = mean
        self.std = std
        self.inplace = inplace
        if isinstance(std, numbers.Number) and std == 0:
            raise ValueError("std cannot be 0")
        if isinstance(std, Sequence) and any([s == 0 for s in std]):
            raise ValueError("std {} contained 0 value, cannot be 0".format(std))

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        channel_count = frames.shape[0]
        mean = self._broadcast_to_seq(self.mean, channel_count)
        std = self._broadcast_to_seq(self.std, channel_count)
        return VF.normalize(frames, mean, std, inplace=self.inplace)

    @staticmethod
    def _broadcast_to_seq(
        x: Union[numbers.Number, Sequence], channel_count: int
    ) -> Sequence[numbers.Number]:
        if isinstance(x, numbers.Number):
            return [x] * channel_count
        # else assume already a sequence
        return x

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(mean={mean}, std={std})".format(
            mean=self.mean, std=self.std
        )


class ResizeVideo:
    """Resize the input video (composed of PIL Images) to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            ``(h, w)``, output size will be matched to this. If size is an ``int``,
            smaller edge of the image will be matched to this number.
            i.e, if ``height > width``, then image will be rescaled to
            ``(size * height / width, size)``.
        interpolation (int, optional): Desired interpolation. Default is
            :py:const:`PIL.Image.BILINEAR` (see :py:meth:`PIL.Image.Image.resize` for
            other options).
    """

    def __init__(
        self, size: Union[Tuple[int, int], int], interpolation=PIL.Image.BILINEAR
    ):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, frames: Iterator[Image]) -> Iterator[Image]:
        for frame in frames:
            yield F.resize(frame, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = T._pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + "(size={0!r}, interpolation={1})".format(
            self.size, interpolate_str
        )


class MultiScaleCropVideo:
    r"""Random crop the input video (composed of PIL Images) at one of the given
    scales or from a set of fixed crops, then resize to specified size.



    Args:
        size (sequence or int): Desired output size. If size is an
            int instead of sequence like ``(w, h)``, a square image ``(size, size)`` is
            made.
        scales (sequence): A sequence of floats between in the range :math:`[0, 1]`
            indicating the scale of the crop to be made.
        max_distortion (int): Integer between 0--``len(scales)`` that controls
            aspect-ratio distortion. This parameters decides which scales will be
            combined together when creating crop boxes. A max distortion of ``0``
            means that the crop width/height have to be from the same scale, whereas a
            distortion of 1 means that the crop width/height can be from 1 scale
            before or ahead in the ``scales`` sequence thereby stretching or squishing
            the frame.
        fixed_crops (bool): Whether to use upper right, upper left, lower right,
            lower left and center crop positions as the list of candidate crop positions
            instead of those generated from ``scales`` and ``max_distortion``.
        more_fixed_crops (bool): Whether to add center left, center right, upper center,
            lower center, upper quarter left, upper quarter right, lower quarter left,
            lower quarter right crop positions to the list of candidate crop
            positions that are randomly selected. ``fixed_crops`` must be enabled to use
            this setting.
    """

    def __init__(
        self,
        size,
        scales: Tuple[float] = (1, 0.875, 0.75, 0.66),
        max_distortion: int = 1,
        fixed_crops: bool = True,
        more_fixed_crops: bool = True,
    ):
        if isinstance(size, numbers.Number):
            self.size = ImageShape(int(size), int(size))
        else:
            self.size = ImageShape(*size)
        self.scales = scales
        self.max_distortion = max_distortion
        self.fixed_crops = fixed_crops
        self.more_fixed_crops = more_fixed_crops
        if self.more_fixed_crops and not self.fixed_crops:
            raise ValueError("fixed_crops must be True if using more_fixed_crops.")
        self.interpolation = PIL.Image.BILINEAR

    def __call__(self, frames: Iterator[Image]) -> Iterator[Image]:
        try:
            frames = frames.__iter__()
        except AttributeError:
            pass
        frame = next(frames)

        h, w, i, j = self.get_params(
            frame,
            self.size,
            self.scales,
            max_distortion=self.max_distortion,
            fixed_crops=self.fixed_crops,
            more_fixed_crops=self.more_fixed_crops,
        )
        yield F.resized_crop(
            frame, i, j, h, w, self.size, interpolation=self.interpolation
        )
        for frame in frames:
            yield F.resized_crop(
                frame, i, j, h, w, self.size, interpolation=self.interpolation
            )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(size={size}, scales={scales}, max_distortion={max_distortion},"
            "fixed_crops={fixed_crops}, more_fixed_crops={more_fixed_crops})".format(
                size=self.size,
                scales=self.scales,
                max_distortion=self.max_distortion,
                fixed_crops=self.fixed_crops,
                more_fixed_crops=self.more_fixed_crops,
            )
        )

    @classmethod
    def get_params(
        cls,
        frame: Image,
        output_size: Tuple[int, int],
        scales: Sequence[float],
        max_distortion: int = 0,
        fixed_crops: bool = False,
        more_fixed_crops: bool = False,
    ) -> Tuple[int, int, int, int]:
        output_size = ImageShape(*output_size)
        input_width, input_height = frame.size
        input_shape = ImageShape(input_height, input_width)

        shortest_side_length = min(input_shape)
        crop_sizes = [int(shortest_side_length * scale) for scale in scales]
        crop_shape = cls._sample_crop_shape(crop_sizes, max_distortion, output_size)
        if not fixed_crops:
            offset = cls._sample_random_offset(input_shape, crop_shape)
        else:
            offset = cls._sample_fixed_offset(
                input_shape, crop_shape, more_fixed_crops=more_fixed_crops
            )

        crop_height, crop_width = crop_shape
        h_offset, w_offset = offset
        return crop_height, crop_width, h_offset, w_offset

    @classmethod
    def _sample_crop_shape(cls, crop_sizes, max_distortion, output_shape):
        output_height, output_width = output_shape
        candidate_crop_heights = [
            output_height if abs(crop_size - output_height) < 3 else crop_size
            for crop_size in crop_sizes
        ]
        candidate_crop_widths = [
            output_width if abs(crop_size - output_width) < 3 else crop_size
            for crop_size in crop_sizes
        ]
        crop_shapes = []  # elements of the form: (crop_width, crop_shape)
        for i, crop_height in enumerate(candidate_crop_heights):
            for j, crop_width in enumerate(candidate_crop_widths):
                if abs(i - j) <= max_distortion:
                    crop_shapes.append((crop_height, crop_width))
        return random.choice(crop_shapes)

    @staticmethod
    def _sample_random_offset(input_shape, crop_shape) -> Tuple[int, int]:
        input_height, input_width = input_shape
        crop_height, crop_width = crop_shape
        w_offset = random.randint(0, input_width - crop_width)
        h_offset = random.randint(0, input_height - crop_height)
        return w_offset, h_offset

    @classmethod
    def _sample_fixed_offset(
        cls, input_shape: ImageShape, crop_shape: ImageShape, more_fixed_crops=False
    ) -> Tuple[int, int]:
        offsets = cls._fixed_crop_offsets(
            input_shape, crop_shape, more_fixed_crops=more_fixed_crops
        )
        return random.choice(offsets)

    @staticmethod
    def _fixed_crop_offsets(
        image_shape: ImageShape, crop_shape: ImageShape, more_fixed_crops=False
    ) -> List[Tuple[int, int]]:
        image_h, image_w = image_shape
        crop_h, crop_w = crop_shape
        horizontal_step = (image_w - crop_w) // 4
        vertical_step = (image_h - crop_h) // 4

        # Elements of the form (v_offset, h_offset)
        offsets = [
            (0, 0),  # upper left
            (0, 4 * horizontal_step),  # upper right
            (4 * vertical_step, 0),  # lower left
            (4 * vertical_step, 4 * horizontal_step),  # lower right
            (2 * vertical_step, 2 * horizontal_step),  # center
        ]
        if more_fixed_crops:
            offsets += [
                (2 * vertical_step, 0),  # center left
                (2 * vertical_step, 4 * horizontal_step),  # center right
                (4 * vertical_step, 2 * horizontal_step),  # lower center
                (0 * vertical_step, 2 * horizontal_step),  # upper center
                (1 * vertical_step, 1 * horizontal_step),  # upper left quarter
                (1 * vertical_step, 3 * horizontal_step),  # upper right quarter
                (3 * vertical_step, 1 * horizontal_step),  # lower left quarter
                (3 * vertical_step, 3 * horizontal_step),  # lower right quarter
            ]

        return offsets


class RandomResizedCropVideo:
    """Crop the given video (composed of PIL Images) to random size and aspect ratio.

    A crop of random scale (default: :math:`[0.08, 1.0]`) of the original size and a
    random scale (default: :math:`[3/4, 4/3]`) of the original aspect ratio is
    made. This crop is finally resized to given size. This is popularly used to train
    the Inception networks.

    Args:
        size: expected output size of each edge.
        scale: range of size of the origin size cropped.
        ratio: range of aspect ratio of the origin aspect ratio cropped.
        interpolation: Default: :py:const:`PIL.Image.BILINEAR` (see
            :py:meth:`PIL.Image.Image.resize` for other options).
    """

    def __init__(
        self,
        size: Union[Tuple[int, int], int],
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation=PIL.Image.BILINEAR,
    ):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    def __call__(
        self, frames: Union[Iterator[Image], Iterable[Image]]
    ) -> Iterator[Image]:
        try:
            frames = frames.__iter__()
        except AttributeError:
            pass

        frame = next(frames)
        i, j, h, w = T.RandomResizedCrop.get_params(frame, self.scale, self.ratio)
        yield self._transform_frame(frame, i, j, h, w)
        for frame in frames:
            yield self._transform_frame(frame, i, j, h, w)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(size={size}, interpolation={interpolation}, "
            + "scale={scale}, ratio={ratio}".format(
                size=self.size,
                interpolation=self.interpolation,
                scale=self.scale,
                ratio=self.ratio,
            )
        )

    def _transform_frame(self, frame: Image, i: int, j: int, h: int, w: int) -> Image:
        return F.resized_crop(frame, i, j, h, w, self.size, self.interpolation)


class CollectFrames:
    """Collect frames from iterator into list.

    Used at the end of a sequence of PIL video transformations.
    """

    def __call__(self, frames: Iterator[Image]) -> List[Image]:
        return list(frames)

    def __repr__(self):
        return self.__class__.__name__ + "()"


class PILVideoToTensor:
    r"""Convert a list of PIL Images to a tensor :math:`(C, T, H, W)`."""

    def __init__(self, rescale=True):
        """
        Args:
            rescale: Whether or not to rescale video from :math:`[0, 255]` to
                :math:`[0, 1]`. If ``False`` the tensor will be in range
                :math:`[0, 255]`.
        """
        self.rescale = rescale

    def __call__(self, frames: Union[Iterable[Image], Iterator[Image]]) -> torch.Tensor:
        # PIL Images are in the format (H, W, C)
        # F.to_tensor converts (H, W, C) to (C, H, W)
        # Since we have a list of these tensors, when we stack them we get shape
        # (T, C, H, W), we want to swap T and C to get (C, T, H, W)
        if isinstance(frames, Iterator):
            frames = list(frames)
        tensor = torch.stack(list(map(F.to_tensor, frames))).transpose(1, 0)
        # torchvision.transforms.functional.to_tensor rescales by default, so if the
        # rescaling is disabled we effectively have to invert the operation.
        if not self.rescale:
            tensor *= 255
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + "()"


class NDArrayToPILVideo:
    """Convert :py:class:`numpy.ndarray` of the format :math:`(T, H, W, C)` or :math:`(
    C, T, H, W)` to a PIL video (an iterator of PIL images)
    """

    def __init__(self, format="thwc"):
        """

        Args:
            format: dimensional layout of array, one of ``"thwc"`` or ``"cthw"``
        """
        if format.lower() not in {"thwc", "cthw"}:
            raise ValueError(
                "Invalid format {!r}, only 'thwc' and 'cthw' are "
                "supported".format(format)
            )
        self.format = format

    def __call__(self, frames: np.ndarray) -> Iterator[Image]:
        if self.format == "cthw":
            frames = np.moveaxis(frames, 0, -1)

        for frame in frames:
            yield PIL.Image.fromarray(frame)

    def __repr__(self):
        return self.__class__.__name__ + "(format={!r})".format(self.format)


class TimeApply:
    """Apply a PIL Image transform across time.

    See :std:doc:`torchvision/transforms` for suitable *deterministic*
    transforms to use with meta-transform.

    .. warning:: You should only use this with deterministic image transforms. Using a
       transform like :class:`torchvision.transforms.RandomCrop` will randomly crop
       each individual frame at a different location producing a nonsensical video.

    """

    def __init__(self, img_transform: Callable[[Image], Image]) -> None:
        """
        Args:
            img_transform: Image transform operating on a PIL Image.
        """
        self.img_transform = img_transform

    def __call__(
        self, frames: Union[Iterable[Image], Iterator[Image]]
    ) -> Iterator[Image]:
        for frame in frames:
            yield self.img_transform(frame)


class TimeToChannel:
    r"""Combine time dimension into the channel dimension by reshaping video tensor of
    shape :math:`(C, T, H, W)` into :math:`(C \times T, H, W)`
    """

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        return VF.time_to_channel(frames)

    def __repr__(self):
        return self.__class__.__name__ + "()"
