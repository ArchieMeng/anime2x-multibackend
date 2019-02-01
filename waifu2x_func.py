import argparse
import os
import sys
import time

import chainer
import numpy as np
from PIL import Image
import six

programDir = os.path.dirname(__file__)
waifu2xPath = os.path.join(programDir, "waifu2x-chainer")
sys.path.append(waifu2xPath)
from lib import iproc
from lib import reconstruct
from lib import srcnn
from lib import utils


DEBUG = False


def debug_print(*args, **kwargs):
    if DEBUG:
        six.print_(*args, **kwargs)


def denoise_image(cfg, src, model):
    dst, alpha = split_alpha(src, model)
    debug_print('Level {} denoising...'.format(cfg.noise_level),
               end=' ', flush=True)
    if cfg.tta:
        dst = reconstruct.image_tta(
            dst, model, cfg.tta_level, cfg.block_size, cfg.batch_size)
    else:
        dst = reconstruct.image(dst, model, cfg.block_size, cfg.batch_size)
    if model.inner_scale != 1:
        dst = dst.resize((src.size[0], src.size[1]), Image.LANCZOS)
    debug_print('OK')
    if alpha is not None:
        dst.putalpha(alpha)
    return dst


def upscale_image(cfg, src, scale_model, alpha_model=None):
    dst, alpha = split_alpha(src, scale_model)
    log_scale = np.log2(cfg.scale_ratio)
    for i in range(int(np.ceil(log_scale))):
        debug_print('2.0x upscaling...', end=' ', flush=True)
        model = alpha_model
        if i == 0 or alpha_model is None:
            model = scale_model
        if model.inner_scale == 1:
            dst = iproc.nn_scaling(dst, 2)  # Nearest neighbor 2x scaling
            alpha = iproc.nn_scaling(alpha, 2)  # Nearest neighbor 2x scaling
        if cfg.tta:
            dst = reconstruct.image_tta(
                dst, model, cfg.tta_level, cfg.block_size, cfg.batch_size)
        else:
            dst = reconstruct.image(dst, model, cfg.block_size, cfg.batch_size)
        if alpha_model is None:
            alpha = reconstruct.image(
                alpha, scale_model, cfg.block_size, cfg.batch_size)
        else:
            alpha = reconstruct.image(
                alpha, alpha_model, cfg.block_size, cfg.batch_size)
        debug_print('OK')
    dst_w = int(np.round(src.size[0] * cfg.scale_ratio))
    dst_h = int(np.round(src.size[1] * cfg.scale_ratio))
    if np.round(log_scale % 1.0, 6) != 0 or log_scale <= 0:
        debug_print('Resizing...', end=' ', flush=True)
        dst = dst.resize((dst_w, dst_h), Image.LANCZOS)
        debug_print('OK')
    if alpha is not None:
        if alpha.size[0] != dst_w or alpha.size[1] != dst_h:
            alpha = alpha.resize((dst_w, dst_h), Image.LANCZOS)
        dst.putalpha(alpha)
    return dst


def split_alpha(src, model):
    alpha = None
    if src.mode in ('L', 'RGB', 'P'):
        if isinstance(src.info.get('transparency'), bytes):
            src = src.convert('RGBA')
    rgb = src.convert('RGB')
    if src.mode in ('LA', 'RGBA'):
        debug_print('Splitting alpha channel...', end=' ', flush=True)
        alpha = src.split()[-1]
        rgb = iproc.alpha_make_border(rgb, alpha, model)
        debug_print('OK')
    return rgb, alpha


def load_models(cfg):
    ch = 3 if cfg.color == 'rgb' else 1
    if cfg.model_dir is None:
        model_dir = os.path.join(waifu2xPath, 'models/{}').format(cfg.arch.lower())
    else:
        model_dir = cfg.model_dir

    models = {}
    flag = False
    if cfg.method == 'noise_scale':
        model_name = 'anime_style_noise{}_scale_{}.npz'.format(
            cfg.noise_level, cfg.color)
        model_path = os.path.join(model_dir, model_name)
        if os.path.exists(model_path):
            models['noise_scale'] = srcnn.archs[cfg.arch](ch)
            chainer.serializers.load_npz(model_path, models['noise_scale'])
            alpha_model_name = 'anime_style_scale_{}.npz'.format(cfg.color)
            alpha_model_path = os.path.join(model_dir, alpha_model_name)
            models['alpha'] = srcnn.archs[cfg.arch](ch)
            chainer.serializers.load_npz(alpha_model_path, models['alpha'])
        else:
            flag = True
    if cfg.method == 'scale' or flag:
        model_name = 'anime_style_scale_{}.npz'.format(cfg.color)
        model_path = os.path.join(model_dir, model_name)
        models['scale'] = srcnn.archs[cfg.arch](ch)
        chainer.serializers.load_npz(model_path, models['scale'])
    if cfg.method == 'noise' or flag:
        model_name = 'anime_style_noise{}_{}.npz'.format(
            cfg.noise_level, cfg.color)
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            model_name = 'anime_style_noise{}_scale_{}.npz'.format(
                cfg.noise_level, cfg.color)
            model_path = os.path.join(model_dir, model_name)
        models['noise'] = srcnn.archs[cfg.arch](ch)
        chainer.serializers.load_npz(model_path, models['noise'])

    if cfg.gpu >= 0:
        chainer.backends.cuda.check_cuda_available()
        chainer.backends.cuda.get_device(cfg.gpu).use()
        for _, model in models.items():
            model.to_gpu()
    return models
