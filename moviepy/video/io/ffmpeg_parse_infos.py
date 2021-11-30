import os
import re
import subprocess as sp
from dataclasses import dataclass

import structlog
from typing import List, Literal, Tuple

from moviepy.tools import convert_to_seconds
from moviepy.config import FFMPEG_BINARY

logger = structlog.get_logger()

__all__ = ['VideoMetaData', 'ffmpeg_parse_infos']


@dataclass
class VideoMetaData:
    duration: float = None
    size: Tuple[int, int] = None
    video_found: bool = False
    video_fps: float = None
    video_n_frames: int = None
    video_duration: float = None
    video_rotation: int = 0
    video_size: Tuple[int, int] = None

    audio_found: bool = False
    audio_fps: int = 'unknown'

    def __getitem__(self, item):
        """because moviepy expects here a dict so that we should mimic a dict"""
        return getattr(self, item, None)

    def get(self, key, default=None):
        """mimic dict"""
        return getattr(self, key, default)


def ffmpeg_parse_infos(filename, print_infos=False, check_duration=True,
                       fps_source: Literal['fps', 'tbr'] = 'tbr',
                       decode_file=True) -> VideoMetaData:
    """
    Get file infos using ffmpeg.

    Returns a dictionnary with the fields:
    "video_found", "video_fps", "duration", "video_nframes",
    "video_duration", "audio_found", "audio_fps"

    "video_duration" is slightly smaller than "duration" to avoid
    fetching the uncomplete frames at the end, which raises an error.

    """
    _ = decode_file
    assert check_duration
    raw_metadata = get_raw_metadata(filename, print_infos=print_infos)
    lines = raw_metadata.splitlines()

    meta = VideoMetaData()

    meta.duration = parse_duration(is_GIF=filename.endswith('.gif'),
                                   lines=lines)

    lines_video = [l for l in lines
                   if ' Video: ' in l and re.search('\d+x\d+', l)]
    meta.video_found = bool(lines_video)

    if meta.video_found:
        if not meta.duration:
            fps, n_frames = get_alt_fps_n_frames(filename)

            meta.video_duration = n_frames / fps
            meta.video_fps = fps
            meta.video_n_frames = n_frames
            meta.duration = meta.video_duration
        else:
            # we dont use `get_alt_fps_n_frames()` if we found duration from meta.
            # Because the function is heavy to run on long videos.
            meta.video_fps = parse_video_fps(lines_video[0], fps_source)
            meta.video_n_frames = int(meta.duration * meta.video_fps) + 1
            meta.video_duration = meta.duration

        meta.video_size = parse_video_size(lines_video[0])
        meta.video_fps = adjust_fps(meta.video_fps)

        # We could have also recomputed the duration from the number
        # of frames, as follows:
        # >>> result['video_duration'] = result['video_nframes'] / result['video_fps']

        meta.video_rotation = parse_rotation(lines)

    meta.audio_found, meta.audio_fps = parse_audio_infos(lines)
    return meta


def parse_video_fps(line: str, fps_source: Literal['fps', 'tbr']) -> float:
    # Get the frame rate. Sometimes it's 'tbr', sometimes 'fps', sometimes
    # tbc, and sometimes tbc/2...
    # Current policy: Trust tbr first, then fps unless fps_source is
    # specified as 'fps' in which case try fps then tbr

    # If result is near from x*1000/1001 where x is 23,24,25,50,
    # replace by x*1000/1001 (very common case for the fps).

    def get_tbr() -> float:
        match = re.search("( [0-9]*.| )[0-9]* tbr", line)

        # Sometimes comes as e.g. 12k. We need to replace that with 12000.
        s_tbr = line[match.start():match.end()].split(' ')[1]
        if "k" in s_tbr:
            tbr = float(s_tbr.replace("k", "")) * 1000
        else:
            tbr = float(s_tbr)
        return tbr

    def get_fps() -> float:
        match = re.search(r" ([0-9.]+)(k?) fps", line)
        fps = float(match.group(1))
        if match.group(2) == 'k':
            fps *= 1e3
        return fps

    fps_by_tbr: float = None
    fps_by_fps: float = None
    try:
        fps_by_tbr = get_tbr()
    except:
        logger.warning("TBR field not found", metadata=line)
    try:
        fps_by_fps = get_fps()
    except:
        logger.warning("FPS field not found", metadata=line)

    if fps_source == 'tbr':
        primary_fps = fps_by_tbr
        fallback_fps = fps_by_fps
    elif fps_source == 'fps':
        primary_fps = fps_by_fps
        fallback_fps = fps_by_tbr
    else:
        raise KeyError(fps_source)

    if primary_fps and primary_fps <= 1e3:
        return primary_fps
    elif fallback_fps and fallback_fps <= 1e3:
        logger.warning(f"Got invalid FPS value: {primary_fps} by key: {fps_source}, use an alternative FPS value",
                       fps=fps_by_fps, tbr=fps_by_tbr)
        return fallback_fps
    else:
        default_fps = 30
        logger.warning(f"Got invalid FPS value by FPS and TBR keys, set default FPS value: {default_fps}",
                       fps=fps_by_fps, tbr=fps_by_tbr)
        return default_fps


def roughly_equal(a, b, epsilon=0.01) -> bool:
    return abs((a / b) - 1) < epsilon


def adjust_fps(fps: float) -> float:
    """
    It is known that a fps of 24 is often written as 24000/1001
    but then ffmpeg nicely rounds it to 23.98, which we hate.
    """
    coef = 1000.0 / 1001.0
    for x in [23, 24, 25, 30, 50]:
        if (fps != x) and roughly_equal(fps, x * coef):
            return x * coef
    return fps


def parse_audio_infos(lines: List[str]):
    lines_audio = [l for l in lines if ' Audio: ' in l]

    audio_found = lines_audio != []
    audio_fps = None

    if audio_found:
        line = lines_audio[0]
        try:
            match = re.search(" [0-9]* Hz", line)
            hz_string = line[match.start() + 1:match.end() - 3]  # Removes the 'hz' from the end
            audio_fps = int(hz_string)
        except:
            pass

    return audio_found, audio_fps


def parse_rotation(lines: List[str]) -> int:
    # get the video rotation info.
    rotation_lines = [l for l in lines if 'rotate          :' in l and re.search('\d+$', l)]
    try:
        if len(rotation_lines):
            rotation_line = rotation_lines[0]
            match = re.search('\d+$', rotation_line)
            video_rotation = int(rotation_line[match.start(): match.end()])
        else:
            video_rotation = 0
    except Exception:
        raise IOError(("MoviePy error: failed to read video rotation in file.\n"
                       "Here are the file infos returned by ffmpeg:\n\n%s") % (
                          ''.join(rotation_lines)))
    return video_rotation


def parse_video_size(line: str) -> Tuple[int, int]:
    try:
        # get the size, of the form 460x320 (w x h)
        match = re.search(" [0-9]*x[0-9]*(,| )", line)
        s = line[match.start():match.end() - 1].split('x')
        video_size = (int(s[0]), int(s[1]))
    except Exception as e:
        raise IOError(("MoviePy error: failed to read video dimensions in file.\n"
                       "Here are the file infos returned by ffmpeg:\n\n%s") % (
                          line))
    return video_size


def get_raw_metadata(filename: str, print_infos=False) -> str:
    is_GIF = filename.endswith('.gif')
    cmd = [FFMPEG_BINARY, "-i", filename]
    if is_GIF:
        cmd += ["-f", "null", "/dev/null"]

    infos = call_command(cmd, stream='stderr')

    if print_infos:
        # print the whole info text returned by FFMPEG
        print(infos)

    lines = infos.splitlines()
    if "No such file or directory" in lines[-1]:
        raise IOError(("MoviePy error: the file %s could not be found!\n"
                       "Please check that you entered the correct "
                       "path.") % filename)
    return infos


def parse_duration(is_GIF: bool, lines) -> float:
    keyword = ('frame=' if is_GIF else 'Duration: ')
    # for large GIFS the "full" duration is presented as the last element in the list.
    index = -1 if is_GIF else 0
    line = [l for l in lines if keyword in l][index]
    matches = re.findall("([0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9])", line)
    if matches:
        return convert_to_seconds(matches[0])
    else:
        return 0


def get_alt_fps_n_frames(filename) -> Tuple[float, int]:
    """
    https://superuser.com/a/1179062/643557
    """
    cmd = ["ffprobe", "-show_entries", "stream=r_frame_rate,nb_read_frames",
           "-select_streams", "v", "-count_frames",
           "-of", "compact=p=0:nk=1", "-v", "0", filename]
    out = call_command(cmd, stream='stdout')
    match = re.match(r"(\d+)/(\d+)\|(\d+)", out)  # example: 15/1|166
    fps = int(match.group(1)) / int(match.group(2))
    n_frames = int(match.group(3))
    return fps, n_frames


def call_command(cmd: List[str], stream: Literal['stdout', 'stderr'] = 'stdout') -> str:
    popen_params = {"bufsize": 10 ** 5,
                    "stdout": sp.PIPE,
                    "stderr": sp.PIPE,
                    "stdin": sp.DEVNULL}

    if os.name == "nt":
        popen_params["creationflags"] = 0x08000000

    proc = sp.Popen(cmd, **popen_params)
    (output, error) = proc.communicate()
    if stream == 'stdout':
        result_bytes = output
    elif stream == 'stderr':
        result_bytes = error
    else:
        raise ValueError(stream)
    del proc
    if isinstance(result_bytes, bytes):
        return result_bytes.decode('utf8', errors='replace')
    else:
        return result_bytes


def patch():
    from moviepy.video.io import ffmpeg_reader
    from moviepy.audio.io import readers
    ffmpeg_reader.ffmpeg_parse_infos = ffmpeg_parse_infos
    readers.ffmpeg_parse_infos = ffmpeg_parse_infos
