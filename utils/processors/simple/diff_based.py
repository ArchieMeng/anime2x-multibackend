import time
from multiprocessing import Lock

import numpy as np
from PIL import Image

from . import BaseProcessor
from ..params import ProcessParams


class DiffBasedProcessor(BaseProcessor):
    def __init__(self, params: ProcessParams, postprocessor: BaseProcessor):
        super().__init__(params, postprocessor)
        self._bench_mark_lock = Lock()
        self.last_frame = Image.new(params.input_pix_fmt, (params.input_width, params.input_height))
        self.last_result = Image.new(
            params.input_pix_fmt,
            (int(params.input_width * params.scale), int(params.input_height * params.scale)))
        self.costs = {}
        self.bs_min, self.bs_step = (
            params.additional_params.get('bs_min', 64), params.additional_params.get('bs_step', 64))
        self.bench_epoch = params.additional_params.get('epoch', 5)
        self.threshold = params.additional_params.get('threshold', 0.04)

    def diff_cmp(self, param: np.ndarray):
        return (param * param).sum() / param.size > self.threshold

    def update_processing_timecosts(self):
        ss = min(self.params.input_width, self.params.input_height)
        sl = max(self.params.input_width, self.params.input_height)
        with self._bench_mark_lock:
            # get partial frame process time costs
            for bs in range(self.bs_min, ss + 1, self.bs_step):
                im = Image.new('RGB', (bs, bs))
                t = time.time()
                for _ in range(self.bench_epoch):  # get average cost time
                    im_array = np.array(im)
                    np.array(im) - im_array
                    self.postprocessor.process(im)
                self.costs[bs] = (time.time() - t) / self.bench_epoch

            # get full frame process time cost
            t = time.time()
            im = Image.new('RGB', (sl, sl))
            for _ in range(self.bench_epoch):
                im_array = np.array(im)
                np.array(im) - im_array
                self.postprocessor.process(im)
            self.costs[sl] = (time.time() - t) / self.bench_epoch

    def choose_best_diff_block_size(self, diff: np.ndarray):
        sw, sh = self.params.input_width, self.params.input_height
        best_sz = max(self.params.input_width,
                      self.params.input_height)  # represents all blocks are different, full frame process needed
        best_cnt = 1

        diff_locations = [(0, 0)]  # locations (width-offset, height-offset) of different blocks
        for bs in self.costs.keys():
            bw = bh = bs
            cur_diff_locations = []
            # in case block running out of image borders.
            iw_edge, ih_edge = max(0, sw - bw), max(0, sh - bh)  # max image width and image height
            for iw in map(lambda x: min(iw_edge, x), range(0, sw, bw)):
                for ih in map(lambda x: min(ih_edge, x), range(0, sh, bh)):
                    if self.diff_cmp(diff[ih: ih + bh, iw: iw + bw, :]):
                        cur_diff_locations.append((iw, ih))

            if (cnt := len(cur_diff_locations)) > 0:
                if best_cnt * self.costs[best_sz] > cnt * self.costs[bs]:
                    best_sz = bs
                    best_cnt = cnt
                    diff_locations = cur_diff_locations
            else:  # identical frames, skip
                return bs, []
        return best_sz, diff_locations

    def process(self, im: Image) -> Image:
        im_bytes, pre_fbytes, pre_rbytes = (np.array(im),
                                            np.array(self.last_frame),
                                            np.array(self.last_result))
        diff = im_bytes - pre_fbytes  # difference array calculated directly be minus them
        sw, sh = self.params.input_width, self.params.input_height  # source size (width, height)
        tw, th = (int(self.params.input_width * self.params.scale),
                  int(self.params.input_height * self.params.scale))  # target frame size (width, height)
        bs, diff_locations = self.choose_best_diff_block_size(diff)
        bw = bh = bs
        sl = max(self.params.input_width, self.params.input_height)

        # bypass diff-based method if do full frame processing is faster
        if diff_locations:
            if bw == sl:
                pre_rbytes = np.array(self.postprocessor.process(im))
            else:
                for iw, ih in diff_locations:
                    #  source image width dim destination,
                    #  source image height dim destination
                    sw_des, sh_des = min(iw + bw, sw), min(ih + bh, sh)
                    block = np.array(
                        self.postprocessor
                            .process(im.crop((iw, ih, sw_des, sh_des)))
                            .convert(self.params.input_pix_fmt))
                    pre_rbytes[
                    int(ih * th / sh): int(sh_des * th / sh),
                    int(iw * tw / sw): int(sw_des * tw / sw)
                    ] = block
        result = Image.fromarray(pre_rbytes, self.params.input_pix_fmt)
        self.last_frame = im
        self.last_result = result
        return result
