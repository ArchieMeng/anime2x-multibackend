import os

from PIL import Image

if __package__:
    import importlib

    raw = importlib.import_module(f'{__package__}.waifu2x_ncnn_vulkan_wrapper')
else:
    import waifu2x_ncnn_vulkan_wrapper as raw


class Waifu2x:
    def __init__(self, gpuid=0, model="models-cunet", tta_mode=False, num_threads=1, scale=2, noise=0, tilesize=0):
        self._raw_w2xobj = raw.Waifu2xWrapped(gpuid, tta_mode, num_threads)
        self.model = model
        self.gpuid = gpuid

        self.set_params(scale, noise, tilesize)
        self.load()

    def set_params(self, scale=2, noise=0, tilesize=0):
        """
        set parameters for waifu2x object
        :param scale: 1/2. default: 2
        :param noise: -1/0/1/2. default: -1
        :param tilesize: default: 300
        :return: None
        """
        self._raw_w2xobj.scale = scale
        self._raw_w2xobj.noise = noise
        if tilesize == 0:
            self._raw_w2xobj.tilesize = self.get_tilesize()
        else:
            self._raw_w2xobj.tilesize = tilesize
        self._raw_w2xobj.prepadding = self.get_prepadding()

    def load(self, parampath="", modelpath=""):
        if not parampath or not modelpath:
            if not os.path.isabs(self.model):
                if not os.path.exists(self.model):  # try to load it from module path
                    self.model = os.path.join(os.path.dirname(__file__), "models", self.model)

            if self._raw_w2xobj.noise == -1:
                parampath = os.path.join(self.model, "scale2.0x_model.param")
                modelpath = os.path.join(self.model, "scale2.0x_model.bin")
            elif self._raw_w2xobj.scale == 1:
                parampath = os.path.join(self.model, "noise%d_model.param" % self._raw_w2xobj.noise)
                modelpath = os.path.join(self.model, "noise%d_model.bin" % self._raw_w2xobj.noise)
            elif self._raw_w2xobj.scale == 2:
                parampath = os.path.join(self.model, "noise%d_scale2.0x_model.param" % self._raw_w2xobj.noise)
                modelpath = os.path.join(self.model, "noise%d_scale2.0x_model.bin" % self._raw_w2xobj.noise)

        self._raw_w2xobj.load(parampath, modelpath)

    def process(self, im: Image) -> Image:
        """
        Waifu2x process incoming PIL.Image

        :param im: input PIL.Image
        :return: result PIL.Image
        """
        in_bytes = im.tobytes()
        in_buffer = raw.PixelBuffer(len(in_bytes))
        channels = int(len(in_bytes) / (im.width * im.height))
        out_buffer = raw.PixelBuffer((self._raw_w2xobj.scale ** 2) * len(in_bytes))

        for i, b in enumerate(in_bytes):
            in_buffer[i] = b

        raw_in_image = raw.Image(in_buffer, im.width, im.height, channels)
        raw_out_image = raw.Image(out_buffer, self._raw_w2xobj.scale * im.width, self._raw_w2xobj.scale * im.height,
                                  channels)

        if self.gpuid != -1:
            self._raw_w2xobj.process(raw_in_image, raw_out_image)
        else:
            self._raw_w2xobj.tilesize = max(im.width, im.height)
            self._raw_w2xobj.process_cpu(raw_in_image, raw_out_image)

        out_bytes = bytes(map(lambda i: out_buffer[i], range((self._raw_w2xobj.scale ** 2) * len(in_bytes))))

        return Image.frombytes(im.mode, (self._raw_w2xobj.scale * im.width, self._raw_w2xobj.scale * im.height),
                               out_bytes)

    def get_prepadding(self) -> int:
        if "models-cunet" in self.model:
            if self._raw_w2xobj.noise == -1:
                return 18
            elif self._raw_w2xobj.scale == 1:
                return 28
            elif self._raw_w2xobj.scale == 2:
                return 18
        elif "models-upconv_7_anime_style_art_rgb" in self.model:
            return 7
        elif "models-upconv_7_photo" in self.model:
            return 7
        else:
            raise ValueError('model "%s" is not supported' % self.model)

    def get_tilesize(self):
        if self.gpuid == -1:
            return 4000
        else:
            heap_budget = self._raw_w2xobj.get_heap_budget()
            if 'models-cunet' in self.model:
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

    im = Image.open("../../images/0.jpg")

    t = time()
    w2x_obj = Waifu2x(0, num_threads=8)
    w2x_obj.set_params(noise=0, scale=2)
    w2x_obj.load()
    out_im = w2x_obj.process(im)
    print("Elapsed time: %fs" % (time() - t))
    out_im.save("temp.png")
