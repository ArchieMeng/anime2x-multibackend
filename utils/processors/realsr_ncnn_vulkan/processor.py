from PIL import Image

from .realsr_ncnn_vulkan import RealSR
from ..params import ProcessParams
from ..simple import BaseProcessor


class Processor(BaseProcessor):
    def __init__(self, params: ProcessParams):
        self.upscaler = RealSR(
            params.device_id,
            params.model or "models-DF2K",
            params.tta_mode,
            params.scale,
            params.tilesize,
            params.debug
        )

    def process(self, im: Image) -> Image:
        return self.upscaler.process(im)
