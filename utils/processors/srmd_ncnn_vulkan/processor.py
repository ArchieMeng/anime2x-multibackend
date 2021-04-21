from PIL import Image

if __package__ or '.' in __name__:
    from .srmd_ncnn_vulkan import SRMD
    from ..params import ProcessParams
else:
    from srmd_ncnn_vulkan import SRMD

    ProcessParams = None


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
        return self.upscaler.process(im)
