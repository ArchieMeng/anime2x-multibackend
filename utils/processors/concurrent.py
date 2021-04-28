import datetime
import importlib
import importlib.util
import queue
import time
from collections import deque
from inspect import signature
from multiprocessing import Queue, get_context
from pathlib import Path
from threading import Thread, Event

import ffmpeg
import numpy as np
import six
from PIL import Image

from .params import ProcessParams, FFMPEGParams
from ..utils import print_progress_bar

package_dir = Path(__file__).parent
ctx = get_context()


class SimpleResult:
    def __init__(self):
        self.ready = Event()
        self._result = None

    @property
    def result(self, timeout=None):
        self.ready.wait(timeout)
        return self._result

    @result.setter
    def result(self, obj):
        self._result = obj
        self.ready.set()


class ClearPipeThread(Thread):
    def __init__(self, pipe):
        super().__init__(daemon=True)
        self.pipe = pipe

    def run(self) -> None:
        while self.pipe.read(1024):
            self.pipe.read(1024)


class SetResultThread(Thread):
    def __init__(self, result_dict: dict, result_q: Queue):
        super().__init__(daemon=True)
        self.result_dict = result_dict
        self.result_q = result_q

    def run(self) -> None:
        while True:
            task_result = self.result_q.get()
            if task_result:
                task_id, result = task_result
                self.result_dict[task_id].result = result
                del self.result_dict[task_id]
                del result
            elif self.result_dict:
                continue
            else:
                return


class FrameReaderThread(Thread):
    def __init__(self, video_info, in_q: Queue, result_q: Queue, out_q: queue.Queue, params: ProcessParams):
        super().__init__(daemon=True)
        self.in_q = in_q
        self.result_q = result_q
        self.out_q = out_q
        self.params = params
        self.result_dict = {}
        self.queue_count = 0
        SetResultThread(self.result_dict, result_q).start()

        backend = importlib.import_module(f"{__package__}.{self.params.backend}")

        self.buffer_size = len(signature(backend.Processor.process).parameters) - 1
        self.buffer = deque()

        input_name = video_info['file']['filename']
        self.decoder = (
            ffmpeg
                .input(input_name)['v']
                .output('pipe:',
                        format='rawvideo',
                        pix_fmt='rgb24',
                        r=params.original_frame_rate,
                        vsync='1')
                .run_async(pipe_stdout=True,
                           quiet=(not self.params.debug))
        )

    def run(self) -> None:
        while True:
            b = self.decoder.stdout.read(3 * self.params.input_width * self.params.input_height)
            if not b:
                self.in_q.put(None)
                self.out_q.put(None)
                return

            im = Image.frombytes('RGB', (self.params.input_width, self.params.input_height), b)
            self.buffer.append(im)

            # release bytes as soon as possible
            del b

            self.result_dict[self.queue_count] = result = SimpleResult()
            self.in_q.put((self.queue_count, tuple(self.buffer)))

            if len(self.buffer) == self.buffer_size:  # keep buffer size within buffer_size
                self.buffer.popleft()

            # release image as soon as possible
            del im

            self.out_q.put(result)
            self.queue_count += 1
            del result


class FrameWriterThread(Thread):
    def __init__(self, video_info, q: queue.Queue, params: FFMPEGParams):
        super().__init__(daemon=True)
        self.q = q
        self.params = params

        self.total_size = round(float(video_info['file']['duration']) * params.frame_rate)
        input_name = video_info['file']['filename']
        streams = [(ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', r=params.frame_rate, vsync='1',
                                 s='{}x{}'.format(params.width, params.height))), ]
        if 'audio' in video_info:
            streams.append(ffmpeg.input(input_name)['a'])

        if 'subtitle' in video_info:
            streams.append(ffmpeg.input(input_name)['s'])

        self.encoder = (
            ffmpeg
                .output(*streams,
                        params.filepath,
                        pix_fmt=params.pix_fmt,
                        strict='experimental',
                        vcodec=params.vcodec,
                        acodec=params.acodec,
                        r=params.frame_rate,
                        crf=params.crf,
                        vsync='1',
                        **params.additional_params)
                .overwrite_output()
                .run_async(pipe_stdin=True,
                           quiet=(not params.debug))
        )

    def run(self) -> None:
        cnt = 0
        total_size = self.total_size
        print_progress_bar(0, total_size, self.params.filepath, "time left: N/A", length=30)

        start = time.time()
        while True:
            result = self.q.get()
            if not result:
                self.encoder.stdin.close()
                self.encoder.wait()
                six.print_()
                six.print_("processing time: " + str(datetime.timedelta(0, time.time() - start)))
                return

            for im in result.result:
                b = np.array(im).tobytes()
                self.encoder.stdin.write(b)

            cnt += len(result.result)
            time_cost = (time.time() - start) / cnt
            time_left = str(datetime.timedelta(0, round(time_cost * (total_size - cnt), 0)))

            # release these objects as soon as possible
            del result
            del b

            print_progress_bar(cnt,
                               total_size,
                               self.params.filepath,
                               "time left: {}".format(time_left),
                               decimals=3,
                               length=30)


class WorkerProcess(ctx.Process):
    def __init__(self, params: ProcessParams, in_q: Queue, out_q: Queue, **kwargs):
        super().__init__(**kwargs)
        self.in_q = in_q
        self.out_q = out_q
        self.params = params
        self.processor = None

    def run(self) -> None:
        backend = importlib.import_module(f"{__package__}.{self.params.backend}")
        self.processor = backend.Processor(self.params)
        if self.params.diff_based:
            from .simple.diff_based import DiffBasedProcessor
            self.processor = DiffBasedProcessor(self.params, self.processor)

        while True:
            task = self.in_q.get()
            if task:
                task_id, ims = task
                if self.params.debug:
                    six.print_(f"{self.name}: processing {task_id}")

                ims = self.processor.process(*ims)
                if self.params.debug:
                    six.print_(f"{self.name}: {task_id} done")

                if not isinstance(ims, (tuple, list)):
                    ims = (ims,)

                self.out_q.put((task_id, ims))

                # delete im obj as soon as possible
                del ims
            else:
                # notify other processes
                self.in_q.put(None)
                # notify result setting thread
                self.out_q.put(None)
                return
