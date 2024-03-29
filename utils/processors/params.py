import json
from dataclasses import dataclass, asdict, field
from fractions import Fraction


@dataclass
class ProcessParams:
    backend: str
    input_width: int
    input_height: int
    input_pix_fmt: str
    original_frame_rate: Fraction
    frame_rate: Fraction
    device_id: int
    model: str
    scale: float
    denoise_level: int
    debug: bool
    tilesize: int
    additional_args: list = field(default_factory=list)
    n_threads: int = 1
    diff_based: bool = False
    tta_mode: bool = False

    def to_json(self):
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(cls, s: str):
        # Todo: implement from json method or from yaml method
        pass


@dataclass
class FFMPEGParams:
    '''
    The params for ffmpeg encoders
    '''
    filepath: str
    width: int
    height: int
    pix_fmt: str
    frame_rate: Fraction
    vcodec: str
    acodec: str
    crf: int
    debug: bool
    additional_args: list = field(default_factory=list)
    additional_params: dict = field(default_factory=dict)

    def to_json(self):
        return json.dumps(asdict(self))
