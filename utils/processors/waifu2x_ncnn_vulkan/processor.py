from PIL import Image

if __package__ or '.' in __name__:
    from .waifu2x_ncnn_vulkan import Waifu2x
    from ..params import ProcessParams
else:
    from waifu2x_ncnn_vulkan import Waifu2x
    ProcessParams = None


class Processor:
    def __init__(self, params: ProcessParams):
        # do this in multiprocessing context
        if __package__ or '.' in __name__:
            from .waifu2x_ncnn_vulkan import Waifu2x
        else:
            from waifu2x_ncnn_vulkan import Waifu2x

        self.w2x = Waifu2x(gpuid=params.device_id,
                           model=params.model or "models-cunet",
                           scale=params.scale,
                           noise=params.denoise_level,
                           tilesize=max(params.tilesize, 0),
                           tta_mode=params.tta_mode,
                           num_threads=max(params.n_threads, 1))

    def process(self, im: Image):
        return self.w2x.process(im)
