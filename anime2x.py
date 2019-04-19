import datetime
import importlib
import os
import sys
import time

import argparse
import ffmpeg
import six
from PIL import Image
from pymediainfo import MediaInfo

import utils.utils as ut

# noinspection PyCompatibility

programDir = os.path.dirname(__file__)

p = argparse.ArgumentParser()
p.add_argument('--vcodec', default="libx264")
p.add_argument('--acodec', default="copy")
p.add_argument('--input', '-i', default='test.mp4')
p.add_argument('--output_dir', '-o', default='./')
p.add_argument('--extension', '-e', default='mp4')
p.add_argument('--debug', '-D', action='store_true')
p.add_argument('--mute_ffmpeg', action='store_true')
p.add_argument('--mute_waifu2x', action='store_true')
p.add_argument('--diff_based', '-DF', action='store_true')
p.add_argument('--block_size', '-B', type=int, default=400)

# the waifu2x module to use
p.add_argument("--waifu2x", default="waifu2x_chainer")

args, unknown = p.parse_known_args()
waifu2x = importlib.import_module('.'.join(("utils", args.waifu2x)))


def process_video(videoInfo, outputFileName, func, target_size, **kwargs):
    """
    Process video frames one by one using "func"
    :param videoInfo:
    :param outputFileName:
    :param func:
    :return:
    """

    width, height = videoInfo['Video']['width'], videoInfo['Video']['height']
    cnt, total_size = 0, (videoInfo['General']['duration']
                          * float(videoInfo['General']['frame_rate'])
                          / 1000)
    name = videoInfo['General']['complete_name']
    tmpFileName = (videoInfo['General']['file_name']
                   + '_tmp.'
                   + outputFileName.split('.')[-1])

    process1 = (
        ffmpeg
            .input(name)['v']
            .output('pipe:',
                    format='rawvideo',
                    pix_fmt='rgb24',
                    r=videoInfo['Video']['frame_rate'])
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
                        pix_fmt='yuv420p',
                        strict='experimental',
                        r=videoInfo['Video']['frame_rate'],
                        **kwargs)
                .overwrite_output()
                .run_async(pipe_stdin=True,
                           pipe_stderr=((not args.debug) or args.mute_ffmpeg))
        )
    else:
        process2 = (
            ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24',
                       s='{}x{}'.format(target_size[0], target_size[1]))
                .output(tmpFileName, pix_fmt='yuv420p',
                        r=videoInfo['Video']['frame_rate'], **kwargs)
                .overwrite_output()
                .run_async(pipe_stdin=True,
                           pipe_stderr=((not args.debug) or args.mute_ffmpeg))
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
        img = img.tobytes()

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
                 outputFileName,
                 pix_fmt='yuv420p',
                 r=videoInfo['Video']['frame_rate'],
                 **kwargs)
         .overwrite_output()
         .run(quiet=True))
        os.remove(tmpFileName)
    else:
        os.rename(tmpFileName, outputFileName)


# mute waifu2x output
waifu2x.DEBUG = args.debug & (not args.mute_waifu2x)

if __name__ == "__main__":
    videoInfo = {track.track_type: track.to_data()
                 for track in MediaInfo.parse(args.input).tracks}
    width, height = videoInfo['Video']['width'], videoInfo['Video']['height']

    # test waifu2x backend, and get the size of target video
    im = waifu2x.process_frame(Image.new("RGB", (width, height)))
    process_frame = waifu2x.process_frame

    # apply diff based process frame function (only process different block)
    if args.diff_based:
        process_frame = ut.get_block_diff_based_process_func(
            (args.block_size, args.block_size),
            (width, height),
            im.size,
            im.mode,
            process_frame
        )

    process_video(videoInfo,
                  ''.join([videoInfo['General']['file_name'],
                           "[processed].",
                           args.extension]),
                  process_frame,
                  im.size,
                  vcodec=args.vcodec,
                  acodec=args.acodec,)
