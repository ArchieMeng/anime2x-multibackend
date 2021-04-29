# anime2x-multibackend

A simple Python program which focus on efficiency in upscaling or/and denoising animes.

It has several advantages:

- It uses Pipe to transfer frames between Main program and backends. So it won't create huge temporary frame image files
  during processing.
- It has what I called "Different-based" method to skip duplicated frame tiles. It has similar idea with
  [Dandere](https://github.com/akai-katto/dandere2x). However, this program will not collect all the unique tiles and
  combining them into one large block for processing. It selects the best tile size, and do the processing on the tiles
  individually which results in a nearly lossless artifact.
- OOP backend design makes it easy to support new backends.

## Requirements

- python >= 3.7
- Chainer
    - (Optional) Cupy (waifu2x-chainer's dependency. For Nvidia GPU support. For those who installed some strange
      versions of CUDA runtime, cupy-cudaXXX might be the correct module to be installed. note: XXX is the version
      number of CUDA runtime. etc. 10.0 -> 100, 7.5 -> 75)
- (Optional) [waifu2x-ncnn-vulkan-python](https://github.com/ArchieMeng/waifu2x-ncnn-vulkan-python) compiled shared
  library. Needed for waifu2x-ncnn-vulkan backend.
- (Optional) [realsr-ncnn-vulkan-python](https://github.com/ArchieMeng/realsr-ncnn-vulkan-python) compiled shared
  library. Needed for realsr-ncnn-vulkan backend.
- (Optional) [srmd-ncnn-vulkan-python](https://github.com/ArchieMeng/srmd-ncnn-vulkan-python) compiled shared library.
  Needed for srmd-ncnn-vulkan backend.
- (Optional) [rife-ncnn-vulkan-python](https://github.com/ArchieMeng/rife-ncnn-vulkan-python) compiled shared library.
  Needed for rife-ncnn-vulkan backend.
- GLIBC >= 2.29, if you are using *-ncnn-vulkan on Linux.
- Pillow
- ffmpeg-python
- ffmpeg

## Supported Backends

- [x] waifu2x-chainer
- [x] waifu2x-ncnn-vulkan
- [x] rife-ncnn-vulkan
- [x] srmd-ncnn-vulkan
- [x] rife-ncnn-vulkan
- [ ] cain-ncnn-vulkan
- [ ] dain-ncnn-vulkan

## Google Colab usages

Go to Google Colab website, and use the **anime2x.ipynb** from the **colab** branch of this repo.

You may need to

- upload videos to your Google Drive and share with a link
- or upload it within the Colab file browser.

## Installation

### Install Python packages

```
pip install -r requirements.txt 
```

### Getting anime2x-chainer

```
git clone --recurse-submodules https://github.com/ArchieMeng/anime2x-multibackend
```

### Testing
```
cd anime2x-multibackend
python anime2x.py
```

## Usage

All backend parameters are set by anime2x. It use universal parameters for all backends.

By default, GPU 0 is used. To use other GPUs or CPUs, specify "-d" option with the device id.

### Examples

#### get help message:

```bash
python anime2x.py -h
usage: anime2x.py [-h] [--vcodec VCODEC] [--acodec ACODEC] [--crf CRF] [--pix_fmt PIX_FMT] [--input INPUT] [--output OUTPUT] [--extension EXTENSION] [--debug]
                  [--diff_based] [--backend BACKEND] [--devices device_id [device_id ...]] [--tilesize TILESIZE] [--denoise DENOISE] [--tta_mode TTA_MODE]
                  [--model MODEL] [--frame_ratio FRAME_RATIO] [--width WIDTH | --height HEIGHT | --scale SCALE]

optional arguments:
  -h, --help            show this help message and exit
  --vcodec VCODEC       The codec of output video stream(s) in ffmpeg
  --acodec ACODEC       The codec of output audio stream(s)
  --crf CRF             CRF setting for video encoding
  --pix_fmt PIX_FMT     pixel format for output video(s)
  --input INPUT, -i INPUT
  --output OUTPUT, -o OUTPUT
                        Output dir or output name
  --extension EXTENSION, -e EXTENSION
                        The extension name of output videos
  --debug, -D
  --diff_based, -DF     Enable difference based processing. In this mode, anime2x will only process changed frames blocks instead of the whole frames
  --backend BACKEND, -b BACKEND
                        The backend module to use. By default, waifu2x-ncnn-vulkan is used
  --devices device_id [device_id ...], -d device_id [device_id ...]
                        The device(s) to use. -N for CPU, etc. -1 for 1 CPU and -8 for 8 CPUs. device_id >= 0 represents the related GPU device. 0 for GPU 0 and 1
                        for GPU 1.
  --tilesize TILESIZE, -t TILESIZE
  --denoise DENOISE, -n DENOISE
  --tta_mode TTA_MODE
  --model MODEL, -m MODEL
  --frame_ratio FRAME_RATIO, -f FRAME_RATIO
  --width WIDTH, -W WIDTH
  --height HEIGHT, -H HEIGHT
  --scale SCALE, -s SCALE
```

#### GPU 0 mode with input, scale and denoise level.

```bash
python anime2x.py -d 0 -i test.mp4 -s 2 -n 1 
```

- -d, --devices <device_id ..>  : devices to use. -1 for CPU mode in one thread and -8 for 8 threads. id >= 0 for GPU
  devices, etc. 0 for GPU 0.
- -i, --input \<path>    : the input video file or directory path.
- -s, --scale \[ float ]   : the scale ratio.
- -n --noise_level \[ int ]    : denoise strength

#### simply upscale video and output with video codec HEVC

```bash
python anime2x.py -i video.mkv --vcodec libx265
```

#### upscale video and output with two GPU 0 worker processes

```bash
python anime2x.py -i video.mkv -d 0 0
```

#### upscale video to 1080p

```bash
python anime2x.py -i video.mkv -H 1080
```

#### use waifu2x-ncnn-vulkan backend

```bash
python anime2x.py -i video.mkv --backend waifu2x-ncnn-vulkan
```

#### use rife-ncnn-vulkan backend to smooth video

```bash
python anime2x.py -i video.mkv -s 1 -f 2 --backend rife-ncnn-vulkan
```

* **"-f/--frame_ratio 1"** and **"-s/--scale 1"** are currently required for interpolation

However, to use
waifu2x-ncnn-vulkan, [waifu2x-ncnn-vulkan-python](https://github.com/ArchieMeng/waifu2x-ncnn-vulkan-python)
should be compiled. This is not included in this project. To do so, either refer to the build guide
on [waifu2x-ncnn-vulkan-python](https://github.com/ArchieMeng/waifu2x-ncnn-vulkan-python) or download release archive,
and copy the "models, waifu2x_ncnn_vulkan.py, waifu2x_ncnn_vulkan_wrapper.py,_waifu2x_ncnn_vulkan_wrapper.so" files to
utils/processors/waifu2x_ncnn_vulkan directory.

#### using difference based method in upscaling / denoising

```bash
python anime2x.py -DF -i video.mkv
```

#### Allow debug output such as ffmpeg

```bash
python anime2x.py -D -i video.mkv
```

## How to support other versions of waifu2x?

Write a module which contains a class named "Processor" and place it in the "utils/processors" dir.

### Processor class example:

```python
"""
File: utils/processors/simple/plain.py

This is just an simple example of how Processor looks like.
"""
from PIL import Image

from utils.processors.params import ProcessParams


class Processor:
    def __init__(self, params: ProcessParams, postprocessor=None):
        """
        This processor will return the original frame Image in its process function.
        :param params: parameters for processing. Will be ignored in this class.
        :param postprocessor: the processor that will be used on the result frame before returning it to caller.
        """
        self.params = params
        self.postprocessor = postprocessor

    def process(self, im: Image) -> Image:
        """
        process the frame image
        :param im:
        :return:
        """
        pass  # do some process here
        if self.postprocessor:
            im = self.postprocessor.process(im)
        return im
```

## Benchmarks

Environment: GTX 1050Ti, python 3.9, Arch Linux

#### File: [アニメ『ドールズフロントライン』ティザーPV／Anime[Girls' Frontline]teaser PV](https://www.youtube.com/watch?v=Bx5y9iwblpA) (Downscaled to 540p for benchmark)

|backend|models|devices|Use DF|Timing|
|---|---|---|---|---|
|waifu2x-chainer|UpResNet10|GPU 0|Yes|0:10:32|
|waifu2x-chainer|UpResNet10|GPU 0|No|0:12:21|
|waifu2x-chainer|UpResNet10|GPU 0,GPU 0|Yes|0:09:13|
|waifu2x-chainer|UpResNet10|GPU 0,GPU 0|No|0:11:02|
|waifu2x-ncnn-vulkan|Cunet|GPU 0|Yes|0:24:07|
|waifu2x-ncnn-vulkan|Cunet|GPU 0|No|0:30:11|

#### File: [Airota&LoliHouse] Horimiya - 09 [WebRip 1080p HEVC-10bit AAC ASSx2].mkv

|backend|models|devices|Use DF|Timing|
|---|---|---|---|---|
|waifu2x-chainer|UpResNet10|GPU 0,GPU 0|Yes|8:36:09|
|waifu2x-chainer|UpResNet10|GPU 0,GPU 0|No|18:04:04|  

## Todo

- [x] support video interpolation
- [ ] fractional frames interpolation ratio supports (Like from 24fps to 60fps)
- [ ] multi-host processing
- [ ] More efficient Video-only super-resolution algorithm

## Known Issues

- run CPU processor along with GPU will slow down the whole process
- run DF on 10bits video may produce incorrect color output
- The default parameters are designed for video superresolution. To use video interpolation, -s 1 -f 2 should be used.