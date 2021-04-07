import argparse
import importlib.util
import os
import sys

import chainer
import numpy as np
import six
from PIL import Image

from ..params import ProcessParams
from ..simple import BaseProcessor

PROJECT_DIR = os.path.dirname(__file__)
waifu2x_path = os.path.join(PROJECT_DIR, "waifu2x-chainer")


def import_waifu2x_module(name):
    spec = importlib.util.spec_from_file_location(
        name,
        os.path.join(waifu2x_path, 'lib', ''.join((name, '.py')))
    )
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo


iproc = import_waifu2x_module("iproc")
reconstruct = import_waifu2x_module("reconstruct")
srcnn = import_waifu2x_module("srcnn")
utils = import_waifu2x_module("utils")
default_model = "UpResNet10"


def debug_print(debug=False, *args, **kwargs):
    if debug:
        six.print_(file=sys.stderr, *args, **kwargs)


def load_models(cfg: ProcessParams, args: argparse.Namespace):
    ch = 3 if cfg.input_pix_fmt.lower() == 'rgb' else 1
    if cfg.model:
        if os.path.isdir(cfg.model):
            model_dir = cfg.model
        else:
            model_dir = os.path.join(waifu2x_path, f'models/{cfg.model.lower()}')
    else:
        cfg.model = default_model
        model_dir = os.path.join(waifu2x_path, f'models/{default_model.lower()}')

    models = {}
    flag = False
    if args.method == 'noise_scale':
        model_name = 'anime_style_noise{}_scale_{}.npz'.format(
            cfg.denoise_level, cfg.input_pix_fmt.lower())
        model_path = os.path.join(model_dir, model_name)
        if os.path.exists(model_path):
            models['noise_scale'] = srcnn.archs[cfg.model](ch)
            chainer.serializers.load_npz(model_path, models['noise_scale'])
            alpha_model_name = 'anime_style_scale_{}.npz'.format(cfg.input_pix_fmt.lower())
            alpha_model_path = os.path.join(model_dir, alpha_model_name)
            models['alpha'] = srcnn.archs[cfg.model](ch)
            chainer.serializers.load_npz(alpha_model_path, models['alpha'])
        else:
            flag = True
    if args.method == 'scale' or flag:
        model_name = 'anime_style_scale_{}.npz'.format(cfg.input_pix_fmt.lower())
        model_path = os.path.join(model_dir, model_name)
        models['scale'] = srcnn.archs[cfg.model](ch)
        chainer.serializers.load_npz(model_path, models['scale'])
    if args.method == 'noise' or flag:
        model_name = 'anime_style_noise{}_{}.npz'.format(
            cfg.denoise_level, cfg.input_pix_fmt.lower())
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            model_name = 'anime_style_noise{}_scale_{}.npz'.format(
                cfg.denoise_level, cfg.input_pix_fmt.lower())
            model_path = os.path.join(model_dir, model_name)
        models['noise'] = srcnn.archs[cfg.input_pix_fmt.lower()](ch)
        chainer.serializers.load_npz(model_path, models['noise'])

    if cfg.device_id >= 0:
        chainer.backends.cuda.check_cuda_available()
        chainer.backends.cuda.get_device(cfg.device_id).use()
        for _, model in models.items():
            model.to_gpu()
    return models


def split_alpha(src, model, debug=False):
    alpha = None
    if src.mode in ('L', 'RGB', 'P') and isinstance(
            src.info.get('transparency'), bytes
    ):
        src = src.convert('RGBA')
    rgb = src.convert('RGB')
    if src.mode in ('LA', 'RGBA'):
        debug_print(debug, 'Splitting alpha channel...', end=' ', flush=True)
        alpha = src.split()[-1]
        rgb = iproc.alpha_make_border(rgb, alpha, model)
        debug_print(debug, 'OK', debug=debug)
    return rgb, alpha


def denoise_image(cfg: ProcessParams, args: argparse.Namespace, src, model):
    dst, alpha = split_alpha(src, model, cfg.debug)
    debug_print(cfg.debug, 'Level {} denoising...'.format(cfg.denoise_level),
                end=' ', flush=True)
    if cfg.tta_mode:
        dst = reconstruct.image_tta(
            dst, model, args.tta_level, cfg.tilesize,
            args.batch_size)
    else:
        dst = reconstruct.image(dst, model, cfg.tilesize, args.batch_size)
    if model.inner_scale != 1:
        dst = dst.resize((src.size[0], src.size[1]), Image.LANCZOS)
    debug_print(cfg.debug, 'OK')
    if alpha is not None:
        dst.putalpha(alpha)
    return dst


def upscale_image(cfg: ProcessParams, args: argparse.Namespace, src, scale_model, alpha_model=None):
    dst, alpha = split_alpha(src, scale_model, cfg.debug)
    log_scale = np.log2(cfg.scale)
    for i in range(int(np.ceil(log_scale))):
        debug_print(cfg.debug, '2.0x upscaling...', end=' ', flush=True, )
        model = alpha_model
        if i == 0 or alpha_model is None:
            model = scale_model
        if model.inner_scale == 1:
            dst = iproc.nn_scaling(dst, 2)  # Nearest neighbor 2x scaling
            alpha = iproc.nn_scaling(alpha, 2)  # Nearest neighbor 2x scaling
        if cfg.tta_mode:
            dst = reconstruct.image_tta(dst, model, args.tta_level, cfg.tilesize, args.batch_size)
        else:
            dst = reconstruct.image(dst, model, cfg.tilesize, args.batch_size)
        if alpha_model is None:
            alpha = reconstruct.image(
                alpha, scale_model, cfg.tilesize, args.batch_size)
        else:
            alpha = reconstruct.image(
                alpha, alpha_model, cfg.tilesize, args.batch_size)
        debug_print(cfg.debug, 'OK')
    dst_w = int(np.round(src.size[0] * cfg.scale))
    dst_h = int(np.round(src.size[1] * cfg.scale))
    if np.round(log_scale % 1.0, 6) != 0 or log_scale <= 0:
        debug_print(cfg.debug, 'Resizing...', end=' ', flush=True)
        dst = dst.resize((dst_w, dst_h), Image.LANCZOS)
        debug_print(cfg.debug, 'OK')
    if alpha is not None:
        if alpha.size[0] != dst_w or alpha.size[1] != dst_h:
            alpha = alpha.resize((dst_w, dst_h), Image.LANCZOS)
        dst.putalpha(alpha)
    return dst


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--tta_level', '-T', type=int, default=8,
                   choices=[2, 4, 8])
    p.add_argument('--method', '-m', default='scale',
                   choices=['noise', 'scale', 'noise_scale'])
    p.add_argument('--batch_size', '-b', type=int, default=16)
    return p


class Processor(BaseProcessor):
    def __init__(self, params: ProcessParams):
        p = get_parser()
        self.args = p.parse_args(params.additional_args)

        if params.model and params.model in srcnn.table:
            params.model = srcnn.table[params.model]
        self.models = load_models(params, self.args)
        self.params = params
        if params.tilesize < 32:
            params.tilesize = 128

    def process(self, im: Image) -> Image:
        if 'noise_scale' in self.models:
            return upscale_image(self.params, self.args, im, self.models['noise_scale'], self.models['alpha'])
        if 'noise' in self.models:
            return denoise_image(self.params, self.args, im, self.models['noise'])
        if 'scale' in self.models:
            return upscale_image(self.params, self.args, im, self.models['scale'])
