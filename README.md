# anime2x-chainer
A simple Python program which can upscale or/and denoise animes.
Based on [waifu2x-chainer](https://github.com/tsurumeso/waifu2x-chainer). (But in fact, you can base on other waifu2x implements. All you have to do is just replace the noise_scale with other function which call the function in the waifu2x lib you want)
## Requirements

  - Chainer
  - Cupy (For GPU support. For those who installed some strange versions of CUDA runtime, cupy-cudaXXX might be the correct module to be installed. note: XXX is the version number of CUDA runtime. etc. 10.0 -> 100, 7.5 -> 75)
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
https://github.com/ArchieMeng/anime2x-chainer
```

### Testing
```
cd anime2x-chainer
python main.py
```

## Usage

Almost all the anime processing options are identical to the [waifu2x-chainer](https://github.com/tsurumeso/waifu2x-chainer).

By default, CPU mode with full processes (up to the number of system processors). If you want to use GPU, make sure CUDA and Cupy are installed, and you use '-g' when calling this program. 

### Examples
#### Simple copy test to test program availability
```
$> python main.py --copy_test
main.py:181: RuntimeWarning: the frombuffer defaults may change in a future release; for portability, change the call to read:
  frombuffer(mode, size, data, 'raw', mode, 0, 1)
  img = Image.frombuffer(mode="RGB", data=img, size=(w, h))
main.py:181: RuntimeWarning: the frombuffer defaults may change in a future release; for portability, change the call to read:
  frombuffer(mode, size, data, 'raw', mode, 0, 1)
  img = Image.frombuffer(mode="RGB", data=img, size=(w, h))
main.py:181: RuntimeWarning: the frombuffer defaults may change in a future release; for portability, change the call to read:
  frombuffer(mode, size, data, 'raw', mode, 0, 1)
  img = Image.frombuffer(mode="RGB", data=img, size=(w, h))
main.py:181: RuntimeWarning: the frombuffer defaults may change in a future release; for portability, change the call to read:
  frombuffer(mode, size, data, 'raw', mode, 0, 1)
  img = Image.frombuffer(mode="RGB", data=img, size=(w, h))
test.mp4 |████████████████████████████97.470% time left: 0:00:00.049876
```
This will do nothing to the frames of video. Just for testing encoding/decoding/python function availability.

#### CPU mode with 4 frames processing simultaneously
```bash
python main.py -i video.mkv -j 4
```

#### GPU mode with arch, methods, input, denoise level.
```bash
python main.py -g 0 -a 3 -n 1 -m noise_scale -i test.mp4
```
gpu to use: -g --gpu int
models type: -a --arch ['VGG7', '0', 'UpConv7', '1', 'ResNet10', '2', 'UpResNet10', '3'] 
denoise strength: -n --noise_level [0, 1, 2, 3]
processing method: -m --method ['noise', 'scale', 'noise_scale']
`All the image-related processing options are directly passed to waifu2x-chainer. For more info of such options, please turn to waifu2x-chainer.`

#### simply scale video and output with video codec HEVC & extension 
```bash
python main.py -i video.mkv -m scale --vcodec libx265
```
