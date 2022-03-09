from math import floor

from PIL import Image

try:
    from .srmd_ncnn_vulkan import SRMD
except ImportError:
    from srmd_ncnn_vulkan_python import SRMD

from ..params import ProcessParams


class Processor:
    def __init__(self, params: ProcessParams):
        self.params = params
        self.upscaler = SRMD(max(params.device_id, 0),
                             params.model or "models-srmd",
                             params.tta_mode,
                             params.scale,
                             params.denoise_level,
                             max(params.tilesize, 0), )

    def process(self, im: Image) -> Image:
        w, h = im.size
        cur_scale = 1
        while cur_scale < self.params.scale:
            for scale in range(2, 4):
                if cur_scale * scale >= self.params.scale:
                    cur_scale *= scale
                    self.upscaler.scale = scale
                    im = self.upscaler.process(im)
                    break
            else:
                cur_scale *= 4
                self.upscaler.scale = 4
                im = self.upscaler.process(im)
        w, h = floor(w * self.params.scale), floor(h * self.params.scale)
        im = im.resize((w, h))
        return im
