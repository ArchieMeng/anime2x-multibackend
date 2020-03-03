import argparse
import subprocess
import os
import io
import sys

from PIL import Image

waifu2x_dir = os.path.dirname(__file__) or '.'
args = argparse.ArgumentParser()
args.add_argument("--input",
                  "-i",
                  default="small.png", )

args.add_argument("--output",
                  "-o",
                  default="result.png", )
args.add_argument("--noise_level",
                  "-n",
                  type=int,
                  choices=[-1, 0, 1, 2, 3],
                  default=0)
args.add_argument("--scale_level",
                  "-s",
                  type=int,
                  choices=[1, 2],
                  default=2, )

args.add_argument("--tile_size",
                  "-t",
                  default=128, )

args.add_argument("--model_path",
                  "-m",
                  default="models-cunet")

params, _ = args.parse_known_args()


def process_frame(im: Image, **kwargs) -> Image:
    # Todo enhance system compatibility such as running on windows or linux
    if sys.platform == "linux" or sys.platform == "darwin":
        # Todo test implements on Mac
        # check existence of waifu2x-ncnn-vulkan binary
        p = subprocess.run("which waifu2x-ncnn-vulkan",
                           shell=True, capture_output=True)
        if p.returncode == 0:
            executable = p.stdout[:-1]  # ignore end char '\n'
        else:
            executable = os.path.join(waifu2x_dir,
                                      "waifu2x-ncnn-vulkan")
            if not os.path.exists(executable):
                print("No waifu2x-ncnn-vulkan binary installed or inside utils"
                      " directory. Please install one or compile one and put it"
                      " under utils directory!!!")

        # with waifu2x-ncnn-vulkan installed
        p = subprocess.Popen(args=(executable,
                                   '-i', "/dev/stdin",
                                   '-o', "/dev/stdout",
                                   '-n', str(params.noise_level),
                                   '-s', str(params.scale_level),
                                   '-t', str(params.tile_size),
                                   '-m', params.model_path),
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL,
                             shell=False,
                             cwd=waifu2x_dir)

        bytes_io = io.BytesIO()
        im.save(bytes_io, format="PNG")
        p.stdin.write(bytes_io.getvalue())
        p.stdin.close()
        bytes_io.seek(0)
        bytes_io.write(p.stdout.read())
        im = Image.open(bytes_io)
        return im

    # Todo implement this platform specific parts
    elif sys.platform == "win32":  # Windows CMD
        executable = os.path.join(waifu2x_dir,
                                  "waifu2x-ncnn-vulkan.exe")
        if not os.path.exists(executable):
            print("No waifu2x-ncnn-vulkan.exe binary installed or inside utils"
                  " directory. Please install one or compile one and put it"
                  " under utils directory!!!")
            raise FileNotFoundError(executable + " not found!!")

        temp_filename = "temp.bmp"
        im.save(os.path.join(waifu2x_dir, temp_filename))
        process = subprocess.run(args=(executable,
                             '-i', os.path.join(waifu2x_dir, temp_filename),
                             '-o', os.path.join(waifu2x_dir, temp_filename),
                             '-n', str(params.noise_level),
                             '-s', str(params.scale_level),
                             '-t', str(params.tile_size),
                             '-m', os.path.join(waifu2x_dir, params.model_path)),
                       stderr=subprocess.DEVNULL,
                       shell=False)
        with Image.open(os.path.join(waifu2x_dir, temp_filename)) as temp_im:
            # copy to memory without occupying the temp file on windows
            im = temp_im.copy()

        os.remove(os.path.join(waifu2x_dir, temp_filename))
        return im

    elif sys.platform == "cygwin":  # Windows Cygwin
        raise NotImplementedError()
    else:
        raise NotImplementedError(
            sys.platform + "'s implement is not done yet.")


if __name__ == "__main__":
    im = Image.open("small.png")
    im = process_frame(im)
    im.save('test.png')
