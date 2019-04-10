import datetime
import importlib
import os
import time

import argparse
import ffmpeg
from PIL import Image
from pymediainfo import MediaInfo

import utils.utils as ut

# noinspection PyCompatibility

programDir = os.path.dirname(__file__)

p = argparse.ArgumentParser()
p.add_argument('--scale_ratio', '-s', type=float, default=2.0)
p.add_argument('--input', '-i', default='test.mp4')
p.add_argument('--output_dir', '-o', default='./')
p.add_argument('--extension', '-e', default='mp4')
p.add_argument('--debug', '-d', action='store_true')
p.add_argument('--mute_ffmpeg', action='store_true')
p.add_argument('--mute_waifu2x', action='store_true')
p.add_argument('--vcodec', default="libx264")
p.add_argument('--acodec', default="copy")
p.add_argument("--copy_test", action='store_true')
# the waifu2x module to use
p.add_argument("--waifu2x", default="waifu2x_chainer")
g = p.add_mutually_exclusive_group()
g.add_argument('--width', '-W', type=int, default=0)
g.add_argument('--height', '-H', type=int, default=0)

args, unknown = p.parse_known_args()

waifu2x = importlib.import_module('.'.join(("utils", args.waifu2x)))


def process_video(videoInfo, outputFileName, func, ratio=2.0, **kwargs):
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
                       quiet=(not args.debug) | args.mute_ffmpeg)
    )

    if 'Audio' in videoInfo:
        audioStream = ffmpeg.input(name)['a']
        videoStream = (
            ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24',
                       s='{}x{}'.format(int(width * args.scale_ratio),
                                        int(height * args.scale_ratio)))
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
                           quiet=(not args.debug) | args.mute_ffmpeg)
        )
    else:
        process2 = (
            ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24',
                       s='{}x{}'.format(int(width * args.scale_ratio), int(height * args.scale_ratio)))
                .output(tmpFileName, pix_fmt='yuv420p', r=videoInfo['Video']['frame_rate'], **kwargs)
                .overwrite_output()
                .run_async(pipe_stdin=True, quiet=(not args.debug) | args.mute_ffmpeg)
        )

    ut.print_progress_bar(0, total_size, name, "time left: N/A", length=50)

    time_cost = float(
        'inf')  # this will be used as time_out for result in CPU mode

    start = time.time()
    while True:
        in_bytes = process1.stdout.read(width * height * 3)
        if not in_bytes:
            break

        process2.stdin.write(
            func(in_bytes, width, height)
        )
        cnt += 1
        time_cost = (time.time() - start) / cnt
        time_left = str(datetime.timedelta(0, time_cost * (total_size - cnt)))
        ut.print_progress_bar(cnt,
                              total_size,
                              name,
                              "time left: {}".format(time_left),
                              decimals=3,
                              length=50)

    process2.stdin.close()
    process1.wait()
    process2.wait()

    tmpFileInfo = {track.track_type: track.to_data() for track in MediaInfo.parse(tmpFileName).tracks}
    if tmpFileInfo['Video']['frame_count'] != videoInfo['Video']['frame_count']:
        print("correcting video stream")
        tmpFramesCount, correctFrameCount = tmpFileInfo['Video']['frame_count'], videoInfo['Video']['frame_count']

        audioStream = ffmpeg.input(name)['a']
        videoStream = (
            ffmpeg
                .input(tmpFileName)
                .setpts("{}/{}*PTS".format(correctFrameCount, tmpFramesCount))
        )

        (ffmpeg
                .output(videoStream, audioStream, outputFileName, pix_fmt='yuv420p', r=videoInfo['Video']['frame_rate'],
                        **kwargs)
                .overwrite_output()
                .run())
        os.remove(tmpFileName)
    else:
        os.rename(tmpFileName, outputFileName)


def save_bytes_frame(img, w, h):
    img = Image.frombuffer(mode="RGB", data=img, size=(w, h))
    with open('frame.bytes', 'wb') as fp:
        fp.write(img.tobytes())
    return img.tobytes()


# mute waifu2x output
waifu2x.DEBUG = args.debug & (not args.mute_waifu2x)

if __name__ == "__main__":
    videoInfo = {track.track_type: track.to_data()
                 for track in MediaInfo.parse(args.input).tracks}
    width, height = videoInfo['Video']['width'], videoInfo['Video']['height']

    if args.width != 0:
        args.scale_ratio = args.width / width
    if args.height != 0:
        args.scale_ratio = args.height / height

    if args.copy_test:
        args.scale_ratio = 1.0
        process_video(videoInfo,
                      ''.join([videoInfo['General']['file_name'],
                               "[processed].",
                               args.extension]),
                      save_bytes_frame,
                      args.scale_ratio,
                      vcodec=args.vcodec,
                      acodec=args.acodec, )

    else:
        process_video(videoInfo,
                      ''.join([videoInfo['General']['file_name'],
                               "[processed].",
                               args.extension]),
                      waifu2x.process_frame,
                      args.scale_ratio,
                      vcodec=args.vcodec,
                      acodec=args.acodec, )
