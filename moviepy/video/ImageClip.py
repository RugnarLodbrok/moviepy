import numpy as np
from PIL import ExifTags, Image
from imageio import imread

from moviepy.video.VideoClip import VideoClip
from moviepy.decorators import outplace


def rotate(im: Image.Image) -> Image.Image:
    """
    Some images by IOS could include implicit rotation
    and final video will contain unexpected rotated frames.
    Workaround - apply rotate using meta information.
    """
    exif = im.getexif()
    if not exif:
        return im
    try:
        orientation_key = \
            list(ExifTags.TAGS.keys())[list(ExifTags.TAGS.values()).index('Orientation')]
        key = exif.get(orientation_key, None)
    except (ValueError, IndexError):
        return im
    if key == 3:
        im = im.rotate(180, expand=True)
    elif key == 6:
        im = im.rotate(270, expand=True)
    elif key == 8:
        im = im.rotate(90, expand=True)
    return im


class ImageClip(VideoClip):
    """Class for non-moving VideoClips.

    A video clip originating from a picture. This clip will simply
    display the given picture at all times.

    Examples
    --------

    >>> clip = ImageClip("myHouse.jpeg")
    >>> clip = ImageClip( someArray ) # a Numpy array represent

    Parameters
    ----------

    img
      Any picture file (png, tiff, jpeg, etc.) as a string or a path-like object,
      or any array representing an RGB image (for instance a frame from a VideoClip).

    is_mask
      Set this parameter to `True` if the clip is a mask.

    transparent
      Set this parameter to `True` (default) if you want the alpha layer
      of the picture (if it exists) to be used as a mask.

    Attributes
    ----------

    img
      Array representing the image of the clip.

    """

    def __init__(
        self, img, is_mask=False, transparent=True, fromalpha=False, duration=None
    ):
        VideoClip.__init__(self, is_mask=is_mask, duration=duration)

        if not isinstance(img, np.ndarray):
            # img is a string or path-like object, so read it in from disk
            img = imread(img)

        if len(img.shape) == 3:  # img is (now) a RGB(a) numpy array

            if img.shape[2] == 4:
                if fromalpha:
                    img = 1.0 * img[:, :, 3] / 255
                elif is_mask:
                    img = 1.0 * img[:, :, 0] / 255
                elif transparent:
                    self.mask = ImageClip(1.0 * img[:, :, 3] / 255, is_mask=True)
                    img = img[:, :, :3]
            elif is_mask:
                img = 1.0 * img[:, :, 0] / 255

        # if the image was just a 2D mask, it should arrive here
        # unchanged
        self.make_frame = lambda t: img
        self.size = img.shape[:2][::-1]
        self.img = img

    def transform(self, func, apply_to=None, keep_duration=True):
        """General transformation filter.

        Equivalent to VideoClip.transform. The result is no more an
        ImageClip, it has the class VideoClip (since it may be animated)
        """
        if apply_to is None:
            apply_to = []
        # When we use transform on an image clip it may become animated.
        # Therefore the result is not an ImageClip, just a VideoClip.
        new_clip = VideoClip.transform(
            self, func, apply_to=apply_to, keep_duration=keep_duration
        )
        new_clip.__class__ = VideoClip
        return new_clip

    @outplace
    def image_transform(self, image_func, apply_to=None):
        """Image-transformation filter.

        Does the same as VideoClip.image_transform, but for ImageClip the
        transformed clip is computed once and for all at the beginning,
        and not for each 'frame'.
        """
        if apply_to is None:
            apply_to = []
        arr = image_func(self.get_frame(0))
        self.size = arr.shape[:2][::-1]
        self.make_frame = lambda t: arr
        self.img = arr

        for attr in apply_to:
            a = getattr(self, attr, None)
            if a is not None:
                new_a = a.image_transform(image_func)
                setattr(self, attr, new_a)

    @outplace
    def time_transform(self, time_func, apply_to=None, keep_duration=False):
        """Time-transformation filter.

        Applies a transformation to the clip's timeline
        (see Clip.time_transform).

        This method does nothing for ImageClips (but it may affect their
        masks or their audios). The result is still an ImageClip.
        """
        if apply_to is None:
            apply_to = ["mask", "audio"]
        for attr in apply_to:
            a = getattr(self, attr, None)
            if a is not None:
                new_a = a.time_transform(time_func)
                setattr(self, attr, new_a)


class ColorClip(ImageClip):
    """An ImageClip showing just one color.

    Parameters
    ----------

    size
      Size (width, height) in pixels of the clip.

    color
      If argument ``is_mask`` is False, ``color`` indicates
      the color in RGB of the clip (default is black). If `is_mask``
      is True, ``color`` must be  a float between 0 and 1 (default is 1)

    is_mask
      Set to true if the clip will be used as a mask.

    """

    def __init__(self, size, color=None, is_mask=False, duration=None):
        w, h = size

        if is_mask:
            shape = (h, w)
            if color is None:
                color = 0
            elif not np.isscalar(color):
                raise Exception("Color has to be a scalar when mask is true")
        else:
            if color is None:
                color = (0, 0, 0)
            elif not hasattr(color, "__getitem__"):
                raise Exception("Color has to contain RGB of the clip")
            shape = (h, w, len(color))

        super().__init__(
            np.tile(color, w * h).reshape(shape), is_mask=is_mask, duration=duration
        )