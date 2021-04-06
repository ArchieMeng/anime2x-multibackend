import sys
from math import floor
from pathlib import Path

from Cython import nogil
from PIL import Image

if __package__:
    import importlib

    raw = importlib.import_module(f"{__package__}.waifu2x_ncnn_vulkan_wrapper")
else:
    import waifu2x_ncnn_vulkan_wrapper as raw


class Waifu2x:
    def __init__(
            self,
            gpuid=0,
            model="models-cunet",
            tta_mode=False,
            num_threads=1,
            scale: float = 2,
            noise=0,
            tilesize=0,
    ):
        """
        Waifu2x class which can do image super resolution.

        :param gpuid: the id of the gpu device to use. -1 for cpu mode.
        :param model: the name or the path to the model
        :param tta_mode: whether to enable tta mode or not
        :param num_threads: the number of threads in upscaling
        :param scale: scale level, 1 = no scaling, 2 = upscale 2x. value: float. default: 2
        :param noise: noise level, large value means strong denoise effect, -1 = no effect. value: -1/0/1/2/3. default: 0
        :param tilesize: tile size. 0 for automatically setting the size. default: 0
        """
        self._raw_w2xobj = raw.Waifu2xWrapped(gpuid, tta_mode, num_threads)
        self.model = model
        self.gpuid = gpuid
        self.scale = scale  # the real scale ratio
        self.set_params(scale, noise, tilesize)
        self.load()

    def set_params(self, scale=2., noise=-1, tilesize=0):
        """
        set parameters for waifu2x object

        :param scale: 1/2. default: 2
        :param noise: -1/0/1/2/3. default: -1
        :param tilesize: default: 0
        :return: None
        """
        self._raw_w2xobj.scale = (
            2 if scale > 1 else 1
        )  # control the real scale ratio at each raw process call
        self._raw_w2xobj.noise = noise
        self._raw_w2xobj.tilesize = self.get_tilesize() if tilesize == 0 else tilesize
        self._raw_w2xobj.prepadding = self.get_prepadding()

    def load(self, parampath: str = "", modelpath: str = "") -> None:
        """
        Load models from given paths. Use self.model if one or all of the parameters are not given.

        :param parampath: the path to model params. usually ended with ".param"
        :param modelpath: the path to model bin. usually ended with ".bin"
        :return: None
        """
        if not parampath or not modelpath:
            model_dir = Path(self.model)
            if not model_dir.is_absolute() and (
                    not model_dir.is_dir()):  # try to load it from module path if not exists as directory
                dir_path = Path(__file__).parent
                model_dir = dir_path.joinpath("models", self.model)

            if self._raw_w2xobj.noise == -1:
                parampath = model_dir.joinpath("scale2.0x_model.param")
                modelpath = model_dir.joinpath("scale2.0x_model.bin")
                self._raw_w2xobj.scale = 2
            elif self._raw_w2xobj.scale == 1:
                parampath = model_dir.joinpath(
                    f"noise{self._raw_w2xobj.noise}_model.param"
                )
                modelpath = model_dir.joinpath(
                    f"noise{self._raw_w2xobj.noise}_model.bin"
                )
            elif self._raw_w2xobj.scale == 2:
                parampath = model_dir.joinpath(
                    f"noise{self._raw_w2xobj.noise}_scale2.0x_model.param"
                )
                modelpath = model_dir.joinpath(
                    f"noise{self._raw_w2xobj.noise}_scale2.0x_model.bin"
                )

        if not Path(parampath).exists() or not Path(modelpath).exists():
            raise FileNotFoundError(f"{parampath} or {modelpath} not found")
        parampath_str, modelpath_str = raw.StringType(), raw.StringType()
        if sys.platform in ("win32", "cygwin"):
            parampath_str.wstr = raw.new_wstr_p()
            raw.wstr_p_assign(parampath_str.wstr, str(parampath))
            modelpath_str.wstr = raw.new_wstr_p()
            raw.wstr_p_assign(modelpath_str.wstr, str(modelpath))
        else:
            parampath_str.str = raw.new_str_p()
            raw.str_p_assign(parampath_str.str, str(parampath))
            modelpath_str.str = raw.new_str_p()
            raw.str_p_assign(modelpath_str.str, str(modelpath))

        self._raw_w2xobj.load(parampath_str, modelpath_str)

    def process(self, im: Image) -> Image:
        if self.scale > 1:
            cur_scale = 1
            w, h = im.size
            while cur_scale < self.scale:
                im = self._process(im)
                cur_scale *= 2
            w, h = floor(w * self.scale), floor(h * self.scale)
            im = im.resize((w, h))
        else:
            im = self._process(im)
        return im

    def _process(self, im: Image) -> Image:
        """
        Process the incoming PIL.Image

        :param im: PIL.Image
        :return: PIL.Image
        """
        in_bytes = bytearray(im.tobytes())
        channels = int(len(in_bytes) / (im.width * im.height))
        out_bytes = bytearray((self._raw_w2xobj.scale ** 2) * len(in_bytes))

        raw_in_image = raw.Image(in_bytes, im.width, im.height, channels)
        raw_out_image = raw.Image(
            out_bytes,
            self._raw_w2xobj.scale * im.width,
            self._raw_w2xobj.scale * im.height,
            channels,
        )

        if self.gpuid != -1:
            with nogil:
                self._raw_w2xobj.process(raw_in_image, raw_out_image)
        else:
            self._raw_w2xobj.tilesize = max(im.width, im.height)
            with nogil:
                self._raw_w2xobj.process_cpu(raw_in_image, raw_out_image)

        return Image.frombytes(
            im.mode,
            (self._raw_w2xobj.scale * im.width, self._raw_w2xobj.scale * im.height),
            bytes(out_bytes),
        )

    def get_prepadding(self) -> int:
        if "models-cunet" in self.model:
            if self._raw_w2xobj.noise == -1 or self._raw_w2xobj.scale == 2:
                return 18
            elif self._raw_w2xobj.scale == 1:
                return 28
        elif (
                "models-upconv_7_anime_style_art_rgb" in self.model
                or "models-upconv_7_photo" in self.model
        ):
            return 7
        else:
            raise ValueError(f'model "{self.model}" is not supported')

    def get_tilesize(self):
        if self.gpuid == -1:
            return 4000

        heap_budget = self._raw_w2xobj.get_heap_budget()
        if "models-cunet" in self.model:
            if heap_budget > 2600:
                return 400
            elif heap_budget > 740:
                return 200
            elif heap_budget > 250:
                return 100
            else:
                return 32
        else:
            if heap_budget > 1900:
                return 400
            elif heap_budget > 550:
                return 200
            elif heap_budget > 190:
                return 100
            else:
                return 32


if __name__ == "__main__":
    from time import time

    t = time()
    im = Image.open("../images/0.jpg")
    w2x_obj = Waifu2x(0, noise=0, scale=2)
    out_im = w2x_obj.process(im)
    out_im.save("temp.png")
    print(f"Elapsed time: {time() - t}s")
