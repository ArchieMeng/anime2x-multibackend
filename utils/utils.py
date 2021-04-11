import json
import subprocess
import sys
from pathlib import Path

import six

from .terminalsize import get_terminal_size


def print_progress_bar(iteration,
                       total,
                       prefix='',
                       suffix='',
                       decimals=1,
                       length=100,
                       fill='█'):
    """
    print progress bar that fill the width of terminal. (omit prefix if it is too long)

    :param iteration: Required : current iteration (Int)
    :param total: Required  : total iterations (Int)
    :param prefix: Optional  : prefix string (Str)
    :param suffix: Optional  : suffix string (Str)
    :param decimals: Optional  : positive number of decimals in percent complete (Int)
    :param length: Optional  : character length of bar (Int)
    :param fill: Optional  : bar fill character (Str)
    :return: None
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)

    width, _ = get_terminal_size()
    print_length = len('%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    if print_length > width:
        ignored_length = print_length + 10 - width
        prefix = prefix[:(len(prefix) - ignored_length) // 2] + '....' + prefix[(len(prefix) + ignored_length) // 2:]

    # clear print line before output
    if sys.platform in ['win32', 'cygwin']:
        six.print_(' ' * (width - 1), end='\r')
    else:
        six.print_(' ' * width, end='\r')

    six.print_('%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def get_video_info(video_path):
    # check ffprobe path
    if sys.platform in ["linux", "darwin"]:
        p = subprocess.run('which ffprobe', shell=True, capture_output=True)
        if p.returncode == 0:
            exec_path = p.stdout[:-1]  # ignore ending char '\n'
        else:
            exec_path = str(Path(__file__).parent / "ffprobe")
    elif sys.platform in ['win32', 'cygwin']:  # Todo test on Windows
        p = subprocess.run('ffprobe -h', shell=True, capture_output=True)
        if p.returncode == 0:
            exec_path = "ffprobe"
        else:
            exec_path = str(Path(__file__).parent / "ffprobe")
    else:
        raise NotImplemented(f"{sys.platform} is not supported to call ffprobe in anime2x")

    raw_info = json.loads(subprocess.run(
        (
            exec_path,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "-i",
            video_path,
        ),
        check=False,
        stdout=subprocess.PIPE
    ).stdout)

    if not raw_info:
        return {}

    video_info = {'file': raw_info['format']}
    for stream in raw_info['streams']:
        if stream['codec_type'] in video_info:
            video_info[stream['codec_type']].append(stream)
        else:
            video_info[stream['codec_type']] = [stream]

    video_info['video'] = video_info['video'][0]
    video_info['file']['filename_noext'] = Path(video_info['file']['filename']).name.split('.')[0]
    return video_info
