import json
from dataclasses import dataclass, asdict, field


@dataclass
class ProcessParams:
    backend: str
    input_width: int
    input_height: int
    input_pix_fmt: str
    original_frame_rate: float
    frame_rate: float
    device_id: int
    model: str
    scale: float
    denoise_level: int
    debug: bool
    tilesize: int
    additional_params: dict = field(default_factory=dict)
    n_threads: int = 1
    diff_based: bool = False
    tta_mode: bool = False

    def to_json(self):
        return json.dumps(asdict(self))


@dataclass
class FFMPEGParams:
    '''
    The params for ffmpeg encoders
    '''
    filepath: str
    width: int
    height: int
    pix_fmt: str
    frame_rate: float
    vcodec: str
    acodec: str
    crf: int
    debug: bool
    additional_params: dict = field(default_factory=dict)
