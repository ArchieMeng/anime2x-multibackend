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
    - Pillow
    - pymediainfo
    - ffmpeg-python
    - mediainfo:
        - For Windows platform, mediainfo.dll (Put it under the program root directory or Windows' Path in environmental
          varibles)
        - For other platform, install mediainfo package.

## Supported Backends

- [x] waifu2x-chainer
- [x] waifu2x-ncnn-vulkan
- [ ] rife-ncnn-vulkan (Will come out soon)
- [ ] srmd-ncnn-vulkan (Waiting for subproject srmd-ncnn-vulkan to start)

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

By default, CPU mode is used for waifu2x-chainer and GPU 0 for waifu2x-ncnn-vulkan. If you want to use GPU in
waifu2x-chainer backend, make sure CUDA and Cupy are installed, and you use '-g' with the gpu-id when calling this
program.

### Examples

#### GPU 0 mode with input, scale and denoise level.

```bash
python anime2x.py -d 0 -i test.mp4 -s 2 -n 1 
```

- -d, --devices <device_id ..>  : devices to use. -1 for CPU mode in one thread and -8 for 8 threads. id >= 0 for GPU
  devices, etc. 0 for GPU 0.
- -i, --input \<path>    : the input video file or directory path.
- -s, --scale \[ float ]   : the scale ratio.
- -n --noise_level \[ int ]    : denoise strength

#### simply scale video and output with video codec HEVC & extension

```bash
python anime2x.py -i video.mkv --vcodec libx265
```

#### use waifu2x-ncnn-vulkan backend

```bash
python anime2x.py -i video.mkv --backend waifu2x_ncnn_vulkan
```

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

Write a module which contains "process_frame" function and place it in the "utils/processors" dir.

### Example:

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
        :param postprocessor: the process that will be used on the result frame before returning it to caller.
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

## Todo

- [ ] support video interpolation
- [ ] multi-host processing
- [ ] More efficient Video-only super-resolution algorithm

## Known Issues

- run CPU processor along with GPU will slow down the whole process
- ~~Currently running program without debug option will be stuck during the processing.~~(Fixed on 20210406)