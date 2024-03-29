import argparse
import time
from inspect import signature
from multiprocessing import Lock
from typing import Union

import numpy as np
from PIL import Image

from . import BaseProcessor
from ..params import ProcessParams


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument('--bs_min', default=64, type=int)
    p.add_argument('--epoch', default=5, type=int)
    p.add_argument('--threshold', default=0.04, type=float)
    return p


class DiffBasedProcessor(BaseProcessor):
    def __init__(self, params: ProcessParams, postprocessor):
        super().__init__(params, postprocessor)
        p = get_parser()
        self.args, _ = p.parse_known_args(params.additional_args)
        self._bench_mark_lock = Lock()
        self.buffer_size = len(signature(postprocessor.process).parameters)
        self.last_frame = Image.new(params.input_pix_fmt, (params.input_width, params.input_height))
        self.last_result = tuple(Image.new(params.input_pix_fmt,
                                           (int(params.input_width * params.scale),
                                            int(params.input_height * params.scale)))
                                 for _ in range(int(params.frame_rate / params.original_frame_rate)))
        self.costs = {}
        self.update_processing_timecosts()

    def diff_cmp(self, param: np.ndarray):
        return (param * param).sum() / param.size > self.args.threshold

    def update_processing_timecosts(self):
        ss = min(self.params.input_width, self.params.input_height)  # short edge length of source
        sl = max(self.params.input_width, self.params.input_height)  # long edge length of source

        def dummy_workload(bs: int, _ims=None):
            _ims = _ims or tuple(Image.new('RGB', (bs, bs)) for i in range(self.buffer_size))
            for _ in range(self.args.epoch):  # get average cost time
                im_array = np.array(_ims[0])
                np.array(_ims[0]) - im_array
                self.postprocessor.process(*_ims)

        with self._bench_mark_lock:
            # get partial frame process time costs
            bs = self.args.bs_min
            while bs <= ss:
                t = time.time()
                dummy_workload(bs)
                self.costs[bs] = (time.time() - t) / self.args.epoch
                bs *= 2

            # get full frame process time cost
            t = time.time()
            _ims = tuple(Image.new('RGB', (sl, ss)) for i in range(self.buffer_size))
            dummy_workload(ss, _ims)
            self.costs[sl] = (time.time() - t) / self.args.epoch

    def choose_best_diff_block_size(self, diff: np.ndarray):
        sw, sh = self.params.input_width, self.params.input_height
        best_sz = max(self.params.input_width,
                      self.params.input_height)  # represents all blocks are different, full frame process needed
        best_cnt = 1

        diff_locations = [(0, 0)]  # locations (width-offset, height-offset) of different blocks
        for bs in sorted(self.costs.keys()):
            bw = bh = bs
            cur_diff_locations = []
            # in case block running out of image borders.
            iw_edge, ih_edge = max(0, sw - bw), max(0, sh - bh)  # max image width and image height
            for iw in map(lambda x: min(iw_edge, x), range(0, sw, bw)):
                for ih in map(lambda x: min(ih_edge, x), range(0, sh, bh)):
                    if self.diff_cmp(diff[ih: ih + bh, iw: iw + bw, :]):
                        cur_diff_locations.append((iw, ih))

            cnt = len(cur_diff_locations)
            if cnt > 0:
                if best_cnt * self.costs[best_sz] > cnt * self.costs[bs]:
                    best_sz = bs
                    best_cnt = cnt
                    diff_locations = cur_diff_locations
            else:  # identical frames, skip
                return bs, []
        return best_sz, diff_locations

    def process(self, *ims) -> Union[tuple, list]:
        if len(ims) > 1:  # do current inter frame difference for interpolation tasks
            self.last_frame = ims[0]
            self.last_result = [ims[0]] * len(ims)

        im_bytes, pre_fbytes = np.array(ims[-1]), np.array(self.last_frame)
        pre_rbytes = [np.array(im) for im in self.last_result]
        diff = im_bytes - pre_fbytes  # difference array calculated directly be minus them
        sw, sh = self.params.input_width, self.params.input_height  # source size (width, height)
        bs, diff_locations = self.choose_best_diff_block_size(diff)
        bw = bh = bs
        sl = max(self.params.input_width, self.params.input_height)
        outputs = None

        # bypass diff-based method if do full frame processing is faster
        if diff_locations:
            if bw == sl:
                outputs = self.postprocessor.process(*ims)
                outputs = [im for im in outputs] if isinstance(outputs, tuple) else [outputs]
            else:
                for iw, ih in diff_locations:
                    #  source image width dim destination, source image height dim destination
                    sw_des, sh_des = min(iw + bw, sw), min(ih + bh, sh)
                    blocks = self.postprocessor.process(*(im.crop((iw, ih, sw_des, sh_des)) for im in ims))
                    if not isinstance(blocks, tuple):
                        blocks = [blocks]

                    if not pre_rbytes:
                        pre_rbytes = [np.array(Image.new(
                            self.params.input_pix_fmt,
                            (int(self.params.input_width * self.params.scale),
                             int(self.params.input_height * self.params.scale))))
                            for _ in range(len(blocks))]
                    elif len(pre_rbytes) < len(blocks):
                        n_filling = len(blocks) - len(pre_rbytes)
                        pre_rbytes += [pre_rbytes[-1].copy() for _ in range(n_filling)]

                    for i, block in enumerate(blocks):
                        pre_rbytes[i][
                        int(ih * self.params.scale): int(sh_des * self.params.scale),
                        int(iw * self.params.scale): int(sw_des * self.params.scale)
                        ] = np.array(block)
        else:
            outputs = tuple(self.last_result[-1]
                            for _ in range(int(self.params.frame_rate / self.params.original_frame_rate)))

        self.last_frame = ims[-1]
        self.last_result = outputs or [Image.fromarray(b, self.params.input_pix_fmt) for b in pre_rbytes]
        return self.last_result
