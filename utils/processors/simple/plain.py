"""
This is just an simple example of how Processor looks like.
"""
from PIL import Image

from ..params import ProcessParams


class Processor:
    def __init__(self, params: ProcessParams, postprocessor=None):
        """
        This processor will return the original frame Image in its process function.
        :param params: parameters for processing. Will be ignored in this class.
        :param postprocessor: the process that will be used on the result frame before returning it to caller.
        """
        self.params = params
        self.postprocessor = postprocessor

    def process(self, im: Image) -> Image:
        """
        process the frame image
        :param im:
        :return:
        """
        pass  # do some process here
        if self.postprocessor:
            im = self.postprocessor.process(im)
        return im
