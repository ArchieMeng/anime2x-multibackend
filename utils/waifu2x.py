import argparse
import os

from PIL import Image

from utils.waifu2x_ncnn_vulkan import Waifu2x

waifu2x_dir = os.path.dirname(__file__) or '.'
args = argparse.ArgumentParser()

args.add_argument("--noise",
                  "-n",
                  type=int,
                  choices=[-1, 0, 1, 2, 3],
                  default=-1)
args.add_argument("--scale",
                  "-s",
                  type=int,
                  choices=[1, 2],
                  default=2, )

args.add_argument("--tilesize",
                  "-t",
                  default=0, )

args.add_argument("--model_path",
                  "-m",
                  default="models-cunet")

args.add_argument("--gpu",
                  "-g",
                  type=int,
                  default=0)

args.add_argument("--jobs",
                  "-j",
                  type=int,
                  default=2)

params, _ = args.parse_known_args()

w2x = Waifu2x(gpuid=int(params.gpu),
              model=params.model_path,
              scale=int(params.scale),
              noise=int(params.noise),
              tilesize=int(params.tilesize),
              num_threads=int(params.jobs))


def process_frame(im: Image, **kwargs) -> Image:
    return w2x.process(im)


if __name__ == "__main__":
    im = Image.open("small.png")
    im = process_frame(im)
    im.save('test.png')
