from __future__ import annotations

from argparse import ArgumentParser
from typing import Union

from PIL import Image

from .rife_ncnn_vulkan import RIFE
from ..params import ProcessParams


def get_parser() -> ArgumentParser:
    p = ArgumentParser()
    p.add_argument('--uhd_mode', '-U', action='store_true')
    return p


class Processor:
    def __init__(self, params: ProcessParams):
        p = get_parser()
        self.args, _ = p.parse_known_args(params.additional_args)
        self.interpolator = RIFE(
            gpuid=params.device_id,
            model=params.model or "rife-v2.4",
            scale=int(params.frame_rate / params.original_frame_rate),
            tta_mode=params.tta_mode,
            uhd_mode=self.args.uhd_mode,
            num_threads=params.n_threads,
        )

    def process(self, im0: Image, im1: Image = None) -> Union[Image, tuple]:
        if im1:
            return tuple(self.interpolator.process(im0, im1) + [im1])
        else:
            return im0
