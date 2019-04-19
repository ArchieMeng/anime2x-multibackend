import six
from PIL import Image
import numpy as np


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
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    six.print_('%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def get_block_diff_based_process_func(block_size_,
                                      size_,
                                      target_size,
                                      im_type,
                                      proc_func_, ):
    @static_var(pre_frame=Image.new(im_type, size_),
                pre_result=Image.new(im_type, target_size), )
    def proc(im_: Image):

        # current frame bytes: numpy.array,
        # previous frame bytes: numpy.Array,
        # previous result bytes: numpy.Array
        im_bytes, pre_fbytes, pre_rbytes = (np.array(im_),
                                            np.array(proc.pre_frame),
                                            np.array(proc.pre_result))
        diff = im_bytes - pre_fbytes
        sw, sh = size_  # original size (width, height)
        tw, th = target_size  # target frame size (width, height)
        bw, bh = block_size_  # block size (width, height)

        # in case block running out of image borders.
        iw_edge, ih_edge = max(0, sw - bw), max(0, sh - bh)  # max iw and ih
        for iw in map(lambda x: min(iw_edge, x), range(0, sw, bw)):
            for ih in map(lambda x: min(ih_edge, x), range(0, sh, bh)):

                #  source image width dim destination,
                #  source image height dim destination
                sw_des, sh_des = min(iw + bw, sw), min(ih + bh, sh)

                if diff[ih: ih + bh, iw: iw + bw, :].any():
                    block = np.array(
                        proc_func_(im_.crop((iw, ih, sw_des, sh_des)))
                            .convert(im_type)
                    )
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
    import time

    im = Image.open('small.png')
    # im = Image.open('kazai_original.jpg')
    proc_func = get_block_diff_based_process_func((400, 400),
                                                  im.size,
                                                  tuple(
                                                      2 * x for x in im.size),
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
