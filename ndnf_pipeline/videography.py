import os
import datajoint as dj
import numpy as np
from ndnf_pipeline.utils.pipeline_tools import get_schema_name
from ndnf_pipeline import lab, experiment

schema = dj.schema(get_schema_name('videography'), locals())


@schema
class VideoRecording(dj.Manual):
    """A camera (lab.Device with device_type='camera') recording for a session."""
    definition = """
    -> experiment.Session
    -> lab.Device
    """


@schema
class VideoFile(dj.Manual):
    definition = """
    -> VideoRecording
    video_file_idx  : smallint      # file index within session for this device, ordered chronologically
    ---
    file_path       : varchar(1000) # path relative to dj.config['path.raw_data']
    fps = null      : float         # nominal frames per second (for reference)
    n_frames = null : int           # total number of frames (for reference)
    """


@schema
class VideoFileFrameTimes(dj.Manual):
    """Frame timestamps for a video file — kept separate to avoid loading large blobs unnecessarily."""
    definition = """
    -> VideoFile
    ---
    frame_times : longblob  # (s) timestamp of each frame relative to session start
    """


@schema
class TrialVideo(dj.Computed):
    definition = """
    -> experiment.SessionTrial
    -> VideoRecording
    ---
    video_file_idx    : smallint   # which VideoFile this trial falls in
    trial_start_frame : int        # index of first frame within that file
    trial_end_frame   : int        # index of last frame within that file
    trial_frame_times : longblob   # (s) timestamp of each frame relative to trial start
    """

    @property
    def key_source(self):
        return experiment.SessionTrial * VideoRecording & VideoFileFrameTimes

    def make(self, key):
        trial_start, trial_end = (experiment.SessionTrial & key).fetch1(
            'trial_start_time', 'trial_end_time')
        trial_start = float(trial_start)
        trial_end   = float(trial_end)

        vid_idxs, frame_times_list = (VideoFile * VideoFileFrameTimes & key).fetch(
            'video_file_idx', 'frame_times', order_by='video_file_idx')

        for vid_idx, frame_times in zip(vid_idxs, frame_times_list):
            mask = (frame_times >= trial_start) & (frame_times <= trial_end)
            if not np.any(mask):
                continue
            indices = np.where(mask)[0]
            self.insert1({**key,
                          'video_file_idx':    int(vid_idx),
                          'trial_start_frame': int(indices[0]),
                          'trial_end_frame':   int(indices[-1]),
                          'trial_frame_times': frame_times[mask] - trial_start})
            return
