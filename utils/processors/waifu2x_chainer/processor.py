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


def debug_print(*args, **kwargs):
    if "params" in kwargs:
        debug = kwargs.pop("params").debug
        if debug:
            six.print_(file=sys.stderr, *args, **kwargs)


def load_models(cfg: ProcessParams):
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
    if cfg.additional_params['method'] == 'noise_scale':
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
    if cfg.additional_params['method'] == 'scale' or flag:
        model_name = 'anime_style_scale_{}.npz'.format(cfg.input_pix_fmt.lower())
        model_path = os.path.join(model_dir, model_name)
        models['scale'] = srcnn.archs[cfg.model](ch)
        chainer.serializers.load_npz(model_path, models['scale'])
    if cfg.additional_params['method'] == 'noise' or flag:
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


def split_alpha(src, model, **kwargs):
    alpha = None
    if src.mode in ('L', 'RGB', 'P') and isinstance(
            src.info.get('transparency'), bytes
    ):
        src = src.convert('RGBA')
    rgb = src.convert('RGB')
    if src.mode in ('LA', 'RGBA'):
        debug_print('Splitting alpha channel...', end=' ', flush=True, **kwargs)
        alpha = src.split()[-1]
        rgb = iproc.alpha_make_border(rgb, alpha, model)
        debug_print('OK', **kwargs)
    return rgb, alpha


def denoise_image(cfg: ProcessParams, src, model, **kwargs):
    dst, alpha = split_alpha(src, model, **kwargs)
    debug_print('Level {} denoising...'.format(cfg.denoise_level),
                end=' ', flush=True, **kwargs)
    if cfg.tta_mode:
        dst = reconstruct.image_tta(
            dst, model, cfg.additional_params.get("tta_level", 8), cfg.tilesize,
            cfg.additional_params.get("batch_size", 16))
    else:
        dst = reconstruct.image(dst, model, cfg.tilesize, cfg.additional_params.get("batch_size", 16))
    if model.inner_scale != 1:
        dst = dst.resize((src.size[0], src.size[1]), Image.LANCZOS)
    debug_print('OK', **kwargs)
    if alpha is not None:
        dst.putalpha(alpha)
    return dst


def upscale_image(cfg: ProcessParams, src, scale_model, alpha_model=None, **kwargs):
    dst, alpha = split_alpha(src, scale_model, **kwargs)
    log_scale = np.log2(cfg.scale)
    for i in range(int(np.ceil(log_scale))):
        debug_print('2.0x upscaling...', end=' ', flush=True, **kwargs)
        model = alpha_model
        if i == 0 or alpha_model is None:
            model = scale_model
        if model.inner_scale == 1:
            dst = iproc.nn_scaling(dst, 2)  # Nearest neighbor 2x scaling
            alpha = iproc.nn_scaling(alpha, 2)  # Nearest neighbor 2x scaling
        if cfg.tta_mode:
            dst = reconstruct.image_tta(
                dst, model, cfg.additional_params.get("tta_level", 8), cfg.tilesize,
                cfg.additional_params.get("batch_size", 16))
        else:
            dst = reconstruct.image(dst, model, cfg.tilesize, cfg.additional_params.get("batch_size", 16))
        if alpha_model is None:
            alpha = reconstruct.image(
                alpha, scale_model, cfg.tilesize, cfg.additional_params.get("batch_size", 16))
        else:
            alpha = reconstruct.image(
                alpha, alpha_model, cfg.tilesize, cfg.additional_params.get("batch_size", 16))
        debug_print('OK', **kwargs)
    dst_w = int(np.round(src.size[0] * cfg.scale))
    dst_h = int(np.round(src.size[1] * cfg.scale))
    if np.round(log_scale % 1.0, 6) != 0 or log_scale <= 0:
        debug_print('Resizing...', end=' ', flush=True, **kwargs)
        dst = dst.resize((dst_w, dst_h), Image.LANCZOS)
        debug_print('OK', **kwargs)
    if alpha is not None:
        if alpha.size[0] != dst_w or alpha.size[1] != dst_h:
            alpha = alpha.resize((dst_w, dst_h), Image.LANCZOS)
        dst.putalpha(alpha)
    return dst


class Processor(BaseProcessor):
    def __init__(self, params: ProcessParams):
        if params.model and params.model in srcnn.table:
            params.model = srcnn.table[params.model]
        if params.denoise_level >= 0 and params.scale > 1:
            params.additional_params['method'] = "noise_scale"
        elif params.scale > 1:
            params.additional_params['method'] = "scale"
        else:
            params.additional_params['method'] = "noise"
        self.models = load_models(params)
        self.params = params
        if params.tilesize < 32:
            params.tilesize = 128

    def process(self, im: Image) -> Image:
        if 'noise_scale' in self.models:
            return upscale_image(self.params, im, self.models['noise_scale'], self.models['alpha'], params=self.params)
        if 'noise' in self.models:
            return denoise_image(self.params, im, self.models['noise'], params=self.params)
        if 'scale' in self.models:
            return upscale_image(self.params, im, self.models['scale'], params=self.params)
