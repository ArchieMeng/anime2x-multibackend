import argparse
import sys
import os
import ffmpeg
import time
from pymediainfo import MediaInfo
from PIL import Image
import datetime
import utils

programDir = os.path.dirname(__file__)
sys.path.append(os.path.join(programDir, "waifu2x-chainer"))
import waifu2x



def do_noting_func(*args, **kwargs):
    pass

# mute waifu2x output
waifu2x.six.print_ = do_noting_func

p = argparse.ArgumentParser()
p.add_argument('--gpu', '-g', type=int, default=-1)
p.add_argument('--input', '-i', default='"test.mp4"')
p.add_argument('--output_dir', '-o', default='./')
p.add_argument('--extension', '-e', default='png')
p.add_argument('--quality', '-q', type=int, default=None)
p.add_argument('--arch', '-a',
               choices=['VGG7', '0', 'UpConv7', '1',
                        'ResNet10', '2', 'UpResNet10', '3'],
               default='VGG7')
p.add_argument('--model_dir', '-d', default=None)
p.add_argument('--method', '-m', choices=['noise', 'scale', 'noise_scale'],
               default='scale')
p.add_argument('--scale_ratio', '-s', type=float, default=2.0)
p.add_argument('--noise_level', '-n', type=int, choices=[0, 1, 2, 3],
               default=1)
p.add_argument('--color', '-c', choices=['y', 'rgb'], default='rgb')
p.add_argument('--tta', '-t', action='store_true')
p.add_argument('--tta_level', '-T', type=int, choices=[2, 4, 8], default=8)
p.add_argument('--batch_size', '-b', type=int, default=16)
p.add_argument('--block_size', '-l', type=int, default=128)
g = p.add_mutually_exclusive_group()
g.add_argument('--width', '-W', type=int, default=0)
g.add_argument('--height', '-H', type=int, default=0)

args = p.parse_args()
if args.arch in waifu2x.srcnn.table:
    args.arch = waifu2x.srcnn.table[args.arch]


def process_frames(videoInfo, outputFileName, func, **kwargs):
    """
    Process video frames one by one using "func"
    :param videoInfo:
    :param outputFileName:
    :param func:
    :return:
    """

    width, height = videoInfo['Video']['width'], videoInfo['Video']['height']
    cnt, total_size = 0, videoInfo['General']['duration'] * float(videoInfo['General']['frame_rate']) / 1000
    name = videoInfo['General']['complete_name']

    if args.width != 0:
        args.scale_ratio = args.width / width
    if args.height != 0:
        args.scale_ratio = args.height / height

    process1 = (
        ffmpeg
            .input(name)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True, quiet=True)
    )

    process2 = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24',
                   s='{}x{}'.format(int(width * args.scale_ratio), int(height * args.scale_ratio)))
            .output(outputFileName, pix_fmt='yuv420p', **kwargs)
            .overwrite_output()
            .run_async(pipe_stdin=True, quiet=True)
    )

    print(utils.print_progress_bar(0, total_size, name, "time left: N/A"))

    while True:
        start = time.time()
        in_bytes = process1.stdout.read(width * height * 3)
        if not in_bytes:
            break

        process2.stdin.write(
            func(in_bytes, width, height)
        )
        cnt += 1
        time_cost = time.time() - start
        time_left = str(datetime.timedelta(0, time_cost * (total_size - cnt)))

        # Todo add progress bar
        utils.print_progress_bar(0, total_size, name, "time left: {}".format(time_left), decimals=3, length=50)
        # print('Elapsed time: {:.6f} sec'.format(time_cost), '\t',
        #       "time to finish: {}".format(str(datetime.timedelta(0, time_cost * (total_size - cnt)))), '\t',
        #       "progress: {}%".format(cnt * 100. / total_size))

    process2.stdin.close()
    process1.wait()
    process2.wait()


def noise_scale(img, width, height):
    img = Image.frombytes('RGB', (width, height), img, "raw", "RGB", 0, 1)
    if 'noise_scale' in models:
        img = waifu2x.upscale_image(
            args, img, models['noise_scale'], models['alpha'])
    else:
        if 'noise' in models:
            img = waifu2x.denoise_image(args, img, models['noise'])
        if 'scale' in models:
            img = waifu2x.upscale_image(args, img, models['scale'])
    return img.tobytes()


def save_bytes_frame(img, w, h):
    with open('frame.bytes', 'wb') as fp:
        fp.write(img.tobytes())
    return img


if __name__ == "__main__":
    videoInfo = {track.track_type: track.to_data() for track in MediaInfo.parse(args.input).tracks}
    models = waifu2x.load_models(args)
    process_frames(videoInfo, videoInfo['General']['file_name'] + "[processed].mp4", noise_scale, vcodec="libx265")
    # process_frames(videoInfo, videoInfo['General']['file_name'] + "[processed].mp4", save_bytes_frame)
