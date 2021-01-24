import argparse
import datetime
import importlib
import os
import time

import ffmpeg
import numpy as np
import six
from PIL import Image
from pymediainfo import MediaInfo

import utils.utils as ut

# noinspection PyCompatibility

programDir = os.path.dirname(__file__)

p = argparse.ArgumentParser()
p.add_argument('--vcodec', default="libx264",
               help="The codec of output video stream(s) in ffmpeg")
p.add_argument('--acodec', default="copy",
               help="The codec of output audio stream(s)")
p.add_argument('--crf', default=23,
               help="CRF setting for video encoding")
p.add_argument('--pix_fmt', default="yuv420p",
               help="pixel format for output video(s)")

p.add_argument('--input', '-i', default='test.mp4')
p.add_argument('--output', '-o', default='./',
               help="Output dir or output name")
p.add_argument('--extension', '-e', default='mp4')
p.add_argument('--debug', '-D', action='store_true')
p.add_argument('--mute_ffmpeg', action='store_true')
p.add_argument('--mute_waifu2x', action='store_true')
p.add_argument('--diff_based', '-DF',
               action='store_true',
               help="""Enable difference based processing.
               In this mode, anime2x will only process changed frames blocks
               instead of the whole frames""")
p.add_argument('--block_size', '-B', default=None, type=int,
               help="The size of blocks in difference based operation")

# the waifu2x module to use
p.add_argument("--waifu2x", default="waifu2x_chainer",
               help="""The waifu2x module to use. 
               By default, waifu2x-chainer is used""")

args, unknown = p.parse_known_args()
waifu2x = importlib.import_module('.'.join(("utils", args.waifu2x)))


def process_video(videoInfo, output_, func, target_size, **kwargs):
    """
    Process video frames one by one using "func"
    :param target_size: size of output video file
    :param videoInfo: mediainfo dict of input video file
    :param output_: output path
    :param func: function to process frames
    :return:
    """

    width, height = videoInfo['Video']['width'], videoInfo['Video']['height']
    if 'framerate_num' in videoInfo['Video'] and 'framerate_den' in videoInfo['Video']:
        frame_rate = float(videoInfo['Video']['framerate_num']) / float(videoInfo['Video']['framerate_den'])
    else:
        frame_rate = float(videoInfo['Video']['frame_rate'])
    cnt, total_size = 0, (videoInfo['General']['duration'] * frame_rate / 1000)
    name = videoInfo['General']['complete_name']
    tmpFileName = (str(videoInfo['General']['file_name'])
                   + '_tmp.'
                   + str(output_.split('.')[-1]))

    process1 = (
        ffmpeg
            .input(name)['v']
            .output('pipe:',
                    format='rawvideo',
                    pix_fmt='rgb24',
                    r=frame_rate)
            .run_async(pipe_stdout=True,
                       pipe_stderr=((not args.debug) or args.mute_ffmpeg))
    )

    if 'Audio' in videoInfo:

        audioStream = ffmpeg.input(name)['a']
        videoStream = (
            ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24',
                       s='{}x{}'.format(target_size[0],
                                        target_size[1]))
        )

        process2 = (
            ffmpeg
                .output(videoStream,
                        audioStream,
                        tmpFileName,
                        pix_fmt=args.pix_fmt,
                        strict='experimental',
                        r=frame_rate,
                        **kwargs)
                .overwrite_output()
                .run_async(pipe_stdin=True,
                           quiet=((not args.debug) or args.mute_ffmpeg))
        )
    else:
        process2 = (
            ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24',
                       s='{}x{}'.format(target_size[0], target_size[1]))
                .output(tmpFileName, pix_fmt=args.pix_fmt,
                        r=frame_rate, **kwargs)
                .overwrite_output()
                .run_async(pipe_stdin=True,
                           quiet=((not args.debug) or args.mute_ffmpeg))
        )

    ut.print_progress_bar(0, total_size, name, "time left: N/A", length=30)

    start = time.time()
    while True:
        in_bytes = process1.stdout.read(width * height * 3)
        if not in_bytes:
            break

        img = Image.frombytes('RGB',
                              (width, height),
                              in_bytes,)

        img = func(img)
        img = np.array(img).tobytes()

        process2.stdin.write(img)
        # # ignore process2 output
        # with open('ffmpeg.out', 'a') as fp:
        #     fp.write(process2.stdout.readl(process2.stdout.peek(1)))
        #     fp.write(process2.stderr.readl(process2.stderr.peek(1)))

        cnt += 1
        time_cost = (time.time() - start) / cnt
        time_left = str(datetime.timedelta(0, time_cost * (total_size - cnt)))
        ut.print_progress_bar(cnt,
                              total_size,
                              name,
                              "time left: {}".format(time_left),
                              decimals=3,
                              length=30)

    process2.stdin.close()
    process1.wait()
    process2.wait()
    six.print_()
    six.print_("processing time: " + str(datetime.timedelta(0, time.time() - start)))

    tmpFileInfo = {track.track_type: track.to_data()
                   for track in MediaInfo.parse(tmpFileName).tracks}
    if tmpFileInfo['Video']['frame_count'] != videoInfo['Video']['frame_count']:
        six.print_("syncing video stream and audio stream")
        tmpFramesCount, correctFrameCount = (
        tmpFileInfo['Video']['frame_count'],
        videoInfo['Video']['frame_count'])

        audioStream = ffmpeg.input(name)['a']
        videoStream = (
            ffmpeg
                .input(tmpFileName)
                .setpts("{}/{}*PTS".format(correctFrameCount, tmpFramesCount))
        )

        (ffmpeg
         .output(videoStream,
                 audioStream,
                 output_,
                 pix_fmt=args.pix_fmt,
                 strict='experimental',
                 r=videoInfo['Video']['frame_rate'],
                 **kwargs)
         .overwrite_output()
         .run(quiet=((not args.debug) or args.mute_ffmpeg)))
        os.remove(tmpFileName)
    else:
        os.rename(tmpFileName, output_)


# mute waifu2x output
waifu2x.DEBUG = args.debug & (not args.mute_waifu2x)

if __name__ == "__main__":

    files = []

    if os.path.isdir(args.input):
        input_dir = args.input
        files = os.listdir(args.input)
    else:
        input_dir = '.'
        files.append(args.input)

    output_dir = './'
    output_name = ""
    if os.path.isdir(args.output):
        output_dir = args.output
    else:
        output_name = args.output

    for file in map(lambda x: os.path.join(input_dir, x), files):
        if os.path.isdir(file):  # bypass directory
            continue

        videoInfo = {track.track_type: track.to_data()
                     for track in MediaInfo.parse(file).tracks}
        if 'Video' not in videoInfo:  # bypass non-video files
            continue

        width, height = videoInfo['Video']['width'], \
                        videoInfo['Video']['height']

        # test waifu2x backend, and get the size of target video
        im = waifu2x.process_frame(Image.new("RGB", (width, height)),
                                   dry_run=True)
        process_frame = waifu2x.process_frame

        # apply diff based process frame function
        # (only process different block)
        if args.diff_based:
            from math import gcd
            # use the optimized block size for DF mode but
            # remain a minimal size requirement of 80
            args.block_size = max(args.block_size or gcd(*im.size), 80)

            process_frame = ut.get_block_diff_based_process_func(
                (args.block_size, args.block_size),
                (width, height),
                im.size,
                im.mode,
                process_frame
            )

        if output_name:
            output_path = os.path.join(output_dir, output_name)
        else:
            output_path = os.path.join(output_dir,
                                       ''.join(
                                           [str(videoInfo['General'][
                                                    'file_name']),
                                            '.',
                                            args.extension])
                                       )

        # when output dir is specific and output files exist
        if os.path.exists(output_path) and output_name == "":
            output_path = os.path.join(output_dir, ''.join(
                [str(videoInfo['General']['file_name']),
                 "[{}X{}]".format(im.size[0], im.size[1]), '.',
                 args.extension]))

        process_video(videoInfo,
                      output_path,
                      process_frame,
                      im.size,
                      vcodec=args.vcodec,
                      acodec=args.acodec,
                      crf=args.crf)
