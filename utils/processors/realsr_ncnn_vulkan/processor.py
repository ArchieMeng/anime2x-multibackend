from math import floor

from PIL import Image

try:
    # import from local directory
    from .realsr_ncnn_vulkan import RealSR
except ImportError:
    # import from installed package
    from realsr_ncnn_vulkan_python import RealSR
from ..params import ProcessParams
from ..simple import BaseProcessor


class Processor(BaseProcessor):
    def __init__(self, params: ProcessParams):
        self.params = params
        self.upscaler = RealSR(
            params.device_id,
            params.model or "models-DF2K",
            params.tta_mode,
            4,
            params.tilesize
        )

    def process(self, im: Image) -> Image:
        if self.params.scale > 1:
            cur_scale = 1
            w, h = im.size
            while cur_scale < self.params.scale:
                im = self.upscaler.process(im)
                cur_scale *= 4
            w, h = floor(w * self.params.scale), floor(h * self.params.scale)
            im = im.resize((w, h))

        return im
