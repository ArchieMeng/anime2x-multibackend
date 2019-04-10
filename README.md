# anime2x-multibackend
A simple Python program which can upscale or/and denoise animes.
Based on [waifu2x-chainer](https://github.com/tsurumeso/waifu2x-chainer). (But in fact, you can base on other waifu2x implements. All you have to do is just creating a module which contains a "process_frame" function in utils folder)
## Requirements

  - Chainer
  - (Optional) Cupy (For GPU support. For those who installed some strange versions of CUDA runtime, cupy-cudaXXX might be the correct module to be installed. note: XXX is the version number of CUDA runtime. etc. 10.0 -> 100, 7.5 -> 75)
  - Pillow
  - pymediainfo
  - ffmpeg-python
  - (Required on Windows platform) mediainfo.dll (put it under the program root directory)

## Installation

### Install Python packages
```
pip install -r requirements.txt 
```

### Getting anime2x-chainer
```
git clone --recurse-submodules https://github.com/ArchieMeng/anime2x-chainer
```

### Testing
```
cd anime2x-chainer
python anime2x.py
```

## Usage

Almost all the anime processing options are identical to the [waifu2x-chainer](https://github.com/tsurumeso/waifu2x-chainer).

By default, CPU mode is used. If you want to use GPU, make sure CUDA and Cupy are installed, and you use '-g' when calling this program.

### Examples
#### Simple copy test to test program availability
```
$> python anime2x.py --copy_test
anime2x.py:181: RuntimeWarning: the frombuffer defaults may change in a future release; for portability, change the call to read:
  frombuffer(mode, size, data, 'raw', mode, 0, 1)
  img = Image.frombuffer(mode="RGB", data=img, size=(w, h))
anime2x.py:181: RuntimeWarning: the frombuffer defaults may change in a future release; for portability, change the call to read:
  frombuffer(mode, size, data, 'raw', mode, 0, 1)
  img = Image.frombuffer(mode="RGB", data=img, size=(w, h))
anime2x.py:181: RuntimeWarning: the frombuffer defaults may change in a future release; for portability, change the call to read:
  frombuffer(mode, size, data, 'raw', mode, 0, 1)
  img = Image.frombuffer(mode="RGB", data=img, size=(w, h))
anime2x.py:181: RuntimeWarning: the frombuffer defaults may change in a future release; for portability, change the call to read:
  frombuffer(mode, size, data, 'raw', mode, 0, 1)
  img = Image.frombuffer(mode="RGB", data=img, size=(w, h))
test.mp4 |████████████████████████████97.470% time left: 0:00:00.049876
```
This will do nothing to the frames of video. Just for testing encoding/decoding/python function availability.

#### GPU mode with arch, methods, input, denoise level.
```bash
python anime2x.py -g 0 -a 3 -n 1 -m noise_scale -i test.mp4
```
gpu to use: -g --gpu int

models type: -a --arch ['VGG7', '0', 'UpConv7', '1', 'ResNet10', '2', 'UpResNet10', '3'] 

denoise strength: -n --noise_level [0, 1, 2, 3]

processing method: -m --method ['noise', 'scale', 'noise_scale']

```All the image-related processing options are directly passed to the backend module (such as utils.waifu2x_chainer)```

#### simply scale video and output with video codec HEVC & extension 
```bash
python anime2x.py -i video.mkv -m scale --vcodec libx265
```

## How to support other versions of waifu2x?

Write a module which contain "process_frame" function and place it in the utils folder like utils/waifu2x_chainer.py . 
 
\#1 Function Declaration: process_frame(img: bytes, width: int, height: int,) --> bytes

\#2 Remember, anime2x function use "scale_ratio" or "width" or "length" command line parameters to get the size of upscaled frames. If you use different parameters, please keep up with the previous parameters.

\#3 utils/waifu2x_chainer.py  might be a good example for you.


## Known Issues

1. Running this program with muted ffmpeg output might stick after a period of time.