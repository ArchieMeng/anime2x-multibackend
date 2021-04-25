from argparse import ArgumentParser

from .rife_ncnn_vulkan import RIFE
from ..params import ProcessParams


def get_parser() -> ArgumentParser:
    p = ArgumentParser()
    p.add_argument('--uhd_mode', '-U', action='store_true')
    return p


class Processor:
    def __init__(self, params: ProcessParams):
        p = get_parser()
        self.args = p.parse_args(params.additional_args)
        self.interpolator = RIFE(
            gpuid=params.device_id,
            model=params.model or "rife-HD",
            tta_mode=params.tta_mode,
            uhd_mode=self.args.uhd_mode,
            num_threads=params.n_threads,
        )

    def process(self, im0, im1=None) -> tuple:
        if im1:
            return im0, self.interpolator.process(im0, im1)
        else:
            return im0,
