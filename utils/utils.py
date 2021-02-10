import time

import numpy as np
import six
from PIL import Image

from .terminalsize import get_terminal_size

proc_costs = dict()
bs_min, bs_step = (64, 64)
# Source long edge size, Source short edge size. They will be initialized with real values in importer module(s).
sl, ss = 512, 512


def static_var(**kwargs):
    """
    set static vars for func
    :param kwargs: static vars to be set
    :return: static_var decorator
    """

    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def print_progress_bar(iteration,
                       total,
                       prefix='',
                       suffix='',
                       decimals=1,
                       length=100,
                       fill='█'):
    """
    print progress bar that fill the width of terminal. (omit prefix if it is too long)

    :param iteration: Required : current iteration (Int)
    :param total: Required  : total iterations (Int)
    :param prefix: Optional  : prefix string (Str)
    :param suffix: Optional  : suffix string (Str)
    :param decimals: Optional  : positive number of decimals in percent complete (Int)
    :param length: Optional  : character length of bar (Int)
    :param fill: Optional  : bar fill character (Str)
    :return: None
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)

    width, _ = get_terminal_size()
    print_length = len('%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    if print_length > width:
        ignored_length = print_length + 10 - width
        prefix = prefix[:(len(prefix) - ignored_length) // 2] + '....' + prefix[(len(prefix) + ignored_length) // 2:]

    # os.system(clear_cmd)
    six.print_('%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def get_processing_timecosts(proc_func_) -> dict:
    costs = dict()
    for bs in range(bs_min, ss + 1, bs_step):
        im = Image.new('RGB', (bs, bs))
        im_barray = np.array(im)
        t = time.time()
        np.array(im) - im_barray
        proc_func_(im)
        costs[bs] = time.time() - t
    return costs


def diff_cmp(param: np.ndarray):
    return param.std() > 5


def choose_best_diff_block_size(diff: np.ndarray, size_, proc_func_):
    sw, sh = size_
    best_sz = sl  # represents all blocks are different, full frame process needed
    best_cnt = 1

    diff_locations = [(0, 0)]  # locations (width-offset, height-offset) of different blocks
    for bs in proc_costs.keys():
        bw = bh = bs
        cur_diff_locations = []
        # in case block running out of image borders.
        iw_edge, ih_edge = max(0, sw - bw), max(0, sh - bh)  # max image width and image height
        for iw in map(lambda x: min(iw_edge, x), range(0, sw, bw)):
            for ih in map(lambda x: min(ih_edge, x), range(0, sh, bh)):
                if diff_cmp(diff[ih: ih + bh, iw: iw + bw, :]):
                    cur_diff_locations.append((iw, ih))

        if (cnt := len(cur_diff_locations)) > 0:
            if best_cnt * proc_costs[best_sz] > cnt * proc_costs[bs]:
                best_sz = bs
                best_cnt = cnt
                diff_locations = cur_diff_locations
        else:  # identical frames, skip
            return bs, []
    return best_sz, diff_locations


def get_block_diff_based_process_func(size_,
                                      target_size,
                                      im_type,
                                      proc_func_, ):
    """
    get a wrapped process function that will only process the difference parts between previous frame and current frame

    :param size_: source frame size which is the size of original video
    :param target_size: the size of target output video
    :param im_type: the type of image frame
    :param proc_func_: the processing function to be wrapped
    :return: a wrapped diff-based processing function
    """

    @static_var(pre_frame=Image.new(im_type, size_),
                pre_result=Image.new(im_type, target_size), )
    def proc(im_: Image):
        """
        do the actual process of a frame image

        :param im_: image to be process on
        :return:
        """

        # current frame bytes: numpy.ndarray,
        # previous frame bytes: numpy.ndarray,
        # previous result bytes: numpy.ndarray
        im_bytes, pre_fbytes, pre_rbytes = (np.array(im_),
                                            np.array(proc.pre_frame),
                                            np.array(proc.pre_result))
        diff = im_bytes - pre_fbytes  # difference array calculated directly be minus them
        sw, sh = size_  # source size (width, height)
        tw, th = target_size  # target frame size (width, height)
        bs, diff_locations = choose_best_diff_block_size(diff, size_, proc_func_)
        bw = bh = bs

        # bypass diff-based method if do full frame processing is faster
        if diff_locations:
            if bw == sl:
                pre_rbytes = np.array(proc_func_(im_))
            else:
                for iw, ih in diff_locations:
                    #  source image width dim destination,
                    #  source image height dim destination
                    sw_des, sh_des = min(iw + bw, sw), min(ih + bh, sh)
                    block = np.array(proc_func_(im_.crop((iw, ih, sw_des, sh_des))).convert(im_type))
                    pre_rbytes[
                    int(ih * th / sh): int(sh_des * th / sh),
                    int(iw * tw / sw): int(sw_des * tw / sw)
                    ] = block
        result = Image.fromarray(pre_rbytes, im_type)
        proc.pre_frame = im_
        proc.pre_result = result
        return result

    return proc


if __name__ == "__main__":
    import waifu2x as w2x

    im = Image.open('small.png')
    # im = Image.open('kazai_original.jpg')
    proc_func = get_block_diff_based_process_func(im.size,
                                                  tuple(2 * x for x in im.size),
                                                  im.mode,
                                                  w2x.process_frame)

    t1 = time.time()
    im = proc_func(im)
    im.save("small_2x.png")
    im = proc_func(Image.open("small.png"))
    im.save("small_2x.png")
    t2 = time.time()

    im = w2x.process_frame(Image.open('small.png'))
    im.save("small_2x_org.png")
    im = w2x.process_frame(Image.open("small.png"))
    im.save("small_2x_org.png")
    t3 = time.time()

    six.print_("diff based time cost: %fs" % (t2 - t1))
    six.print_("original time cost: %fs" % (t3 - t2))

    for k, v in get_processing_timecosts(w2x.process_frame).items():
        print(k, '\t', v)
