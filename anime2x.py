import argparse
import os
import queue
import re
from math import floor

import ffmpeg
import six
from pymediainfo import MediaInfo

from utils.processors.concurrent import WorkerProcess, FrameReaderThread, FrameWriterThread, Queue
from utils.processors.params import ProcessParams, FFMPEGParams

programDir = os.path.dirname(__file__)

p = argparse.ArgumentParser()
p.add_argument('--vcodec', default="libx264",
               help="The codec of output video stream(s) in ffmpeg")
p.add_argument('--acodec', default="copy",
               help="The codec of output audio stream(s)")
p.add_argument('--crf', default=23,
               help="CRF setting for video encoding")
p.add_argument('--pix_fmt', default="yuv420p",
               help="pixel format for output video(s)")
p.add_argument('--input', '-i', default='test.mp4')
p.add_argument('--output', '-o', default='./',
               help="Output dir or output name")
p.add_argument('--extension', '-e', default='mp4',
               help="The extension name of output videos")
p.add_argument('--debug', '-D', action='store_true')
p.add_argument('--diff_based', '-DF',
               action='store_true',
               help="""Enable difference based processing.
               In this mode, anime2x will only process changed frames blocks
               instead of the whole frames""")

# the backend module to use
p.add_argument("--backend", '-b', default="waifu2x_ncnn_vulkan",
               help="""The backend module to use. 
               By default, waifu2x-ncnn-vulkan is used""")
p.add_argument('--devices', '-d', default=(-1,),
               nargs="+", type=int, metavar="device_id",
               help="""The device(s) to use.
               -N for CPU, etc. -1 for 1 CPU and -8 for 8 CPUs.
               device_id >= 0 represents the related GPU device. 0 for GPU 0 and 1 for GPU 1.
               """)
p.add_argument('--width', '-W', type=int, default=0)
p.add_argument('--height', '-H', type=int, default=0)
p.add_argument('--tilesize', '-t', type=int, default=0)
p.add_argument('--scale', '-s', type=float, default=2.0)
p.add_argument('--denoise', '-n', type=int, default=-1)
p.add_argument('--tta_mode', type=bool, default=False)
p.add_argument('--model', '-m', type=str, default="")
p.add_argument('--frame_rate', '-f', type=float, default=None)

args, unknown = p.parse_known_args()


def get_framerate(videoInfo):
    if 'framerate_num' in videoInfo['Video'] and 'framerate_den' in videoInfo['Video']:
        return float(videoInfo['Video']['framerate_num']) / float(videoInfo['Video']['framerate_den'])
    else:
        return float(videoInfo['Video']['frame_rate'])


def sync_tmpfile_av(tmp_file_name, video_info, params: FFMPEGParams):
    tmp_file_info = {track.track_type: track.to_data()
                     for track in MediaInfo.parse(tmp_file_name).tracks}
    if 'Video' in tmp_file_info and tmp_file_info['Video']['frame_count'] != video_info['Video']['frame_count']:
        six.print_("syncing video stream and audio stream")
        audio_stream = ffmpeg.input(video_info['General']['complete_name'])['a']
        video_stream = (
            ffmpeg
                .input(tmp_file_name)
                .setpts("{}/{}*PTS".format(video_info['Video']['frame_count'], tmp_file_info['Video']['frame_count']))
        )

        (
            ffmpeg
                .output(video_stream,
                        audio_stream,
                        params.filepath,
                        pix_fmt=args.pix_fmt,
                        strict='experimental',
                        r=video_info['Video']['frame_rate'],
                        **params.additional_params)
                .overwrite_output()
                .run(quiet=(not params.debug))
        )
        os.remove(tmp_file_name)
    else:
        os.rename(tmp_file_name, params.filepath)


def process_video(video_info, processor_params: list[ProcessParams], params: FFMPEGParams):
    """
    Process video frames one by one using processor params
    :param processor_params: the list of ProcessParams for each individual processors
    :param params: the ffmpeg params for encoding
    :param video_info: mediainfo dict of input video file
    :return: None
    """
    tmp_file_name = f"{video_info['General']['file_name']}_tmp.{params.filepath.split('.')[-1]}"
    process_queue = Queue(2 * len(processor_params))
    process_result_queue = Queue()
    encoding_queue = queue.Queue()
    reader = FrameReaderThread(video_info, process_queue, process_result_queue, encoding_queue, processor_params[0])
    encoder = FrameWriterThread(video_info, encoding_queue, params)
    processes = [WorkerProcess(p, process_queue, process_result_queue, daemon=True) for p in processor_params]

    reader.start()
    for process in processes:
        process.start()
    encoder.start()

    reader.join()
    for process in processes:
        process.join()
    encoder.join()

    sync_tmpfile_av(tmp_file_name, video_info, params)


if __name__ == "__main__":

    files = []

    if os.path.isdir(args.input):
        input_dir = args.input
        files = os.listdir(args.input)
    else:
        input_dir = '.'
        files.append(args.input)

    output_dir = './'
    output_name = ""
    if os.path.isdir(args.output):
        output_dir = args.output
    else:
        output_name = args.output

    for file in map(lambda x: os.path.join(input_dir, x), files):
        if os.path.isdir(file):  # bypass directory
            continue

        video_info = {track.track_type: track.to_data()
                      for track in MediaInfo.parse(file).tracks}
        if 'Video' not in video_info:  # bypass non-video files
            continue

        width, height = video_info['Video']['width'], \
                        video_info['Video']['height']

        if args.width != 0:
            args.scale = args.width / width
        if args.height != 0:
            args.scale = args.height / height

        output_width = floor(width * args.scale)
        output_height = floor(height * args.scale)

        process_params = []
        for device_id in args.devices:
            n_threads = abs(device_id) if device_id < 0 else 1
            device_id = max(device_id, -1)

            process_params.append(ProcessParams(
                device_id=device_id,
                backend=re.sub(r'-', '_', args.backend),
                input_width=width,
                input_height=height,
                input_pix_fmt='RGB',
                original_frame_rate=get_framerate(video_info),
                frame_rate=args.frame_rate or get_framerate(video_info),
                debug=args.debug,
                model=args.model,
                scale=args.scale,
                denoise_level=args.denoise,
                tta_mode=args.tta_mode,
                tilesize=args.tilesize,
                n_threads=n_threads,
                diff_based=args.diff_based,
            ))

        if output_name:
            output_path = os.path.join(output_dir, output_name)
        else:
            output_path = os.path.join(output_dir,
                                       ''.join(
                                           [str(video_info['General'][
                                                    'file_name']),
                                            '.',
                                            args.extension])
                                       )

        # when output dir is specific and output files exist
        if os.path.exists(output_path) and output_name == "":
            output_path = os.path.join(output_dir, ''.join(
                [str(video_info['General']['file_name']),
                 "[{}X{}]".format(output_width, output_height), '.',
                 args.extension]))

        process_video(video_info,
                      process_params,
                      FFMPEGParams(
                          width=output_width,
                          height=output_height,
                          filepath=output_path,
                          vcodec=args.vcodec,
                          acodec=args.acodec,
                          crf=args.crf,
                          debug=args.debug,
                          frame_rate=args.frame_rate or get_framerate(video_info),
                          pix_fmt=args.pix_fmt))
