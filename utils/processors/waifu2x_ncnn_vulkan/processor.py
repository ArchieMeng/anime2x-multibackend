from math import floor

from PIL import Image

try:
    from .waifu2x_ncnn_vulkan import Waifu2x
except ImportError:
    from waifu2x_ncnn_vulkan_python import Waifu2x

from ..params import ProcessParams


class Processor:
    def __init__(self, params: ProcessParams):
        self.params = params
        self.w2x = Waifu2x(gpuid=params.device_id,
                           model=params.model or "models-cunet",
                           scale=2 if params.scale > 1 else 1,
                           noise=params.denoise_level,
                           tilesize=max(params.tilesize, 0),
                           tta_mode=params.tta_mode,
                           num_threads=max(params.n_threads, 1))

    def process(self, im: Image):
        if self.params.scale > 1:
            cur_scale = 1
            w, h = im.size
            while cur_scale < self.params.scale:
                im = self.w2x.process(im)
                cur_scale *= 2
            w, h = floor(w * self.params.scale), floor(h * self.params.scale)
            im = im.resize((w, h))
        else:
            im = self.w2x.process(im)
        return im
