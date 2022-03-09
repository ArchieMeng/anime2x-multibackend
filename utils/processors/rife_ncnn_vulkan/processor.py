from __future__ import annotations

from argparse import ArgumentParser
from typing import Union

from PIL import Image

default_model = "rife-v2.3"
try:
    # import from current directory (* the old way)
    from .rife_ncnn_vulkan import RIFE

    default_model = "rife-v2.4"
except ImportError:
    # import from site-packages (* the current way)
    from rife_ncnn_vulkan_python import RIFE
from ..params import ProcessParams


def get_parser() -> ArgumentParser:
    p = ArgumentParser()
    p.add_argument('--uhd_mode', '-U', action='store_true')
    return p


class Processor:
    def __init__(self, params: ProcessParams):
        p = get_parser()
        self.args, _ = p.parse_known_args(params.additional_args)
        self.scale = int(params.frame_rate / params.original_frame_rate)
        self.interpolator = RIFE(
            gpuid=params.device_id,
            model=params.model or default_model,
            scale=2,  # compatible with old version of RIFE bindings
            tta_mode=params.tta_mode,
            uhd_mode=self.args.uhd_mode,
            num_threads=params.n_threads,
        )

    def process(self, im0: Image, im1: Image = None) -> Union[Image, tuple]:
        def _proc(im0: Image, im1: Image, level) -> list[Image]:
            if level == 1:
                return []
            else:
                im = self.interpolator.process(im0, im1)
                if isinstance(im, list):
                    im = im[0]
                level /= 2
                return _proc(im0, im, level) + [im] + _proc(im, im1, level)

        if im1:
            return tuple(_proc(im0, im1, self.scale) + [im1])
        else:
            return im0
