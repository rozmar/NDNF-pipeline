import json
import os
import subprocess
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor, wait as futures_wait
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import datajoint as dj

from ndnf_pipeline.plot.behavior_plots import (
    FILTER_METHODS, _filter_trace, _estimate_sample_interval, ms_to_samples, ms_to_samples_float,
    _get_block_force_axes, _normalize_lut, _pad_lut_to_range)

# Quality presets for the "Generate Video" GUIs. dpi is the dominant cost of rendering (it sets
# how many pixels each frame is rasterized at in _render_frame_chunk's savefig call - the
# CPU-bound step render_trial_video parallelizes across workers), so 'low' renders much faster
# than 'ultra'; bitrate only affects the final ffmpeg encode's file size/quality at that
# resolution. 'ultra' matches what render_trial_video/save_render_params/preview_last_frame
# defaulted to before this preset dropdown existed; the GUIs default to 'low' instead so a quick
# look renders fast, and only step up to 'ultra' for a final/shareable video.
VIDEO_QUALITY_PRESETS = {
    'low':    dict(dpi=50,  bitrate='700k'),
    'medium': dict(dpi=80,  bitrate='1200k'),
    'high':   dict(dpi=115, bitrate='2000k'),
    'ultra':  dict(dpi=150, bitrate='3000k'),
}
VIDEO_QUALITY_DEFAULT = 'low'

_LABELS = {
    'en': dict(time='Time (s)', feedback='Feedback',
               fb_start='start', fb_target='target',
               force_x='Left - Right (g)',
               force_y='Posterior - Anterior (g)',
               force_time='Force (g)', lr_label='Left - Right', pa_label='Posterior - Anterior',
               lbl_target='Target', lbl_current='Current'),
    'hu': dict(time='Idő (mp)', feedback='Visszajelzés',
               fb_start='start', fb_target='cél',
               force_x='Bal - Jobb erő (g)',
               force_y='Hátsó - Elülső erő (g)',
               force_time='Erő (g)', lr_label='Bal - Jobb', pa_label='Hátsó - Elülső',
               lbl_target='Cél', lbl_current='Aktuális'),
}


def _resolve_case_insensitive(root, relative_path):
    """Resolve `relative_path` under `root`, tolerating per-segment case mismatches.

    file_path values are written by Windows/Mac acquisition machines, whose filesystems are
    case-insensitive, so a stored segment like 'behavior' happily refers to a real folder
    named 'Behavior' there. Linux is case-sensitive, so the exact-case join silently resolves
    to a nonexistent path. Falls back to the naive join (so a genuinely-missing file still
    surfaces a normal "not found" rather than a confusing one) when no case-insensitive match
    exists at some segment.
    """
    current = root
    for part in relative_path.split('/'):
        if not part:
            continue
        exact = os.path.join(current, part)
        if os.path.exists(exact):
            current = exact
            continue
        try:
            match = next(e for e in os.listdir(current) if e.lower() == part.lower())
        except (OSError, StopIteration):
            return os.path.join(root, relative_path)
        current = os.path.join(current, match)
    return current


def _camera_video_file(videography, dj, key, camera_name, trials_needed):
    """Resolve a camera's raw video file path + its full per-file frame_times array.

    Trials with no TrialVideo entry for this camera (e.g. a dropped/missing recording, which
    does happen) are skipped; only the first trial that has one is needed to locate the file.
    """
    current_file_idx = None
    video_file_path = None
    for trial in trials_needed:
        tv_query = videography.TrialVideo() & key & {'device': camera_name, 'trial': trial}
        if len(tv_query) != 1:
            continue
        current_file_idx = tv_query.fetch1('video_file_idx')
        vf = (videography.VideoFile() & key & {'device': camera_name,
                                                'video_file_idx': current_file_idx}).fetch1()
        # file_path is recorded with Windows-style backslashes by the acquisition machine;
        # os.path.join/os.path.exists don't treat '\' as a separator on Linux/Mac, so without
        # this the whole thing silently glues into one nonexistent filename with literal
        # backslashes in it (cv2.VideoCapture then just fails to open, with no clear error)
        relative_path = vf['file_path'].replace('\\', '/')
        video_file_path = _resolve_case_insensitive(dj.config['path.raw_data'], relative_path)
        break

    if current_file_idx is None:
        raise RuntimeError(f"No video found for camera '{camera_name}' in any of the requested trials.")

    file_frame_times = (videography.VideoFileFrameTimes() & key & {
        'device': camera_name, 'video_file_idx': current_file_idx}).fetch1('frame_times')
    return video_file_path, file_frame_times


def _camera_frame_window(file_frame_times, t_video_start, t_video_end):
    frame_mask = (file_frame_times >= t_video_start) & (file_frame_times <= t_video_end)
    frame_abs_indices = np.where(frame_mask)[0]         # absolute frame numbers for cv2
    frame_session_times = file_frame_times[frame_mask]  # session-relative timestamps
    return frame_abs_indices, frame_session_times


def load_trial_video_data(subject_id, session, block, trial_start, trial_end, camera_name,
                           camera_name_2=None, pad_start=1.0, pad_end=1.0,
                           filter_method=None, filter_window_ms=50.0, filter_sigma_ms=20.0,
                           filter_polyorder=3):
    """Fetch and normalize everything needed to render a trial-range video for one block.

    If camera_name_2 is given, a second camera's frames are resolved too (clipped to the same
    [t_video_start, t_video_end] window as the first, so both stay in sync) for a side-by-side
    two-camera video; otherwise the video is single-camera as before.

    X is always Left-Right (L<0, R>0) and Y is always Posterior-Anterior (P<0, A>0),
    same convention as ndnf_pipeline.plot.behavior_plots.

    filter_method smooths each trial's force trace (per-trial, same as
    behavior_plots.plot_block_force_figure) with one of FILTER_METHODS: None/'none' (no
    filtering, the default), 'boxcar' (moving average, `filter_window_ms`), 'median'
    (`filter_window_ms`), 'gaussian' (`filter_sigma_ms`), or 'savgol' (Savitzky-Golay
    polynomial fit, `filter_window_ms` and `filter_polyorder`). `filter_window_ms`/
    `filter_sigma_ms` are converted to samples using the block's own force trace sample
    interval, estimated from the first trial. This is the trace drawn in the video dashboard.

    Returns a dict consumed by make_trial_video_frame_figure/preview_last_frame/render_trial_video.
    """
    # imported lazily: importing experiment.py/videography.py at module load time would force
    # a live DataJoint connection as soon as `ndnf_pipeline.plot.videography_plots` is imported
    from ndnf_pipeline import experiment, videography

    key = {'subject_id': subject_id, 'session': session}

    # --- trials ---
    block_trials  = np.sort((experiment.BehaviorTrial() & key & {'block': block}).fetch('trial'))
    trials_needed = block_trials[trial_start: trial_end + 1]

    # --- force LUT and axis info from TaskSettings ---
    task_setting_id      = (experiment.Block() & key & {'block': block}).fetch1('task_setting_id')
    target_force_lut_raw = (experiment.TaskSettings() & key & {'task_setting_id': task_setting_id}).fetch1('target_force_lut')

    # --- normalize to X=LR (L<0, R>0) and Y=PA (P<0, A>0) ---
    # shared with behavior_plots.plot_block_force_figure/plot_session_blocks_overview - was
    # previously reimplemented here with the sign-flip applied to the wrong array axis (flipped
    # dim1 where _normalize_lut flips dim0 for the same lr_idx==0 case, and vice versa), which
    # visibly mirrored the target LUT wrong on any rig where lr_sign/pa_sign is -1; reusing the
    # same function guarantees this stays pixel-consistent with those plots instead of drifting
    force_axes_dict, lr_idx, pa_idx, lr_sign, pa_sign = _get_block_force_axes(
        experiment, subject_id, session, task_setting_id)
    target_force_lut, lut_extent, lr_extent, pa_extent = _normalize_lut(
        target_force_lut_raw, force_axes_dict, lr_idx, pa_idx, lr_sign, pa_sign)

    # --- per-trial data (force, feedback, rewards, threshold crossings) ---
    all_force_t, all_force_0, all_force_1 = [], [], []
    all_reward_t, all_threshold_t, all_lick_t = [], [], []
    all_lickport_t, all_lickport_pos = [], []
    trial_start_times = {}
    trial_periods = []  # per-trial (t_start, t_end, quiescence_end, response_end), all absolute

    # sample interval is estimated once, from the first trial, and reused for every trial in
    # this block (matching estimate_force_sample_interval / plot_block_force_figure) - the
    # acquisition rate doesn't change trial-to-trial, so this avoids re-estimating per trial
    sample_interval_s = None
    lickport_carry = None  # last known lickport position, carried across the trial-to-trial gap
    for trial in trials_needed:
        t_start, t_end = (experiment.SessionTrial() & key & {'trial': trial}).fetch1('trial_start_time', 'trial_end_time')
        t_start, t_end = float(t_start), float(t_end)
        trial_start_times[trial] = t_start

        ft = (experiment.TrialForceTrace()                & key & {'trial': trial}).fetch1('force_trace_time')
        f0 = (experiment.TrialForceTrace.TrialForceAxis() & key & {'trial': trial, 'force_axis_idx': 0}).fetch1('force_trace_value')
        f1 = (experiment.TrialForceTrace.TrialForceAxis() & key & {'trial': trial, 'force_axis_idx': 1}).fetch1('force_trace_value')
        if filter_method and filter_method != 'none':
            if sample_interval_s is None:
                sample_interval_s = _estimate_sample_interval([ft])
            filter_window = ms_to_samples(filter_window_ms, sample_interval_s, minimum=1)
            filter_sigma = ms_to_samples_float(filter_sigma_ms, sample_interval_s, minimum=1e-6)
            f0 = _filter_trace(f0, filter_method, window=filter_window, sigma=filter_sigma, polyorder=filter_polyorder)
            f1 = _filter_trace(f1, filter_method, window=filter_window, sigma=filter_sigma, polyorder=filter_polyorder)
        all_force_t.extend(ft + t_start);  all_force_0.extend(f0);  all_force_1.extend(f1)

        rewards    = np.asarray((experiment.TrialEvent() & key & {'trial': trial, 'trial_event_type': 'reward'}).fetch('trial_event_time'), float)
        thresholds = np.asarray((experiment.TrialEvent() & key & {'trial': trial, 'trial_event_type': 'threshold crossing'}).fetch('trial_event_time'), float)
        go_times   = np.asarray((experiment.TrialEvent() & key & {'trial': trial, 'trial_event_type': 'go'}).fetch('trial_event_time'), float)
        licks      = np.asarray((experiment.TrialEvent() & key & {'trial': trial, 'trial_event_type': 'lick'}).fetch('trial_event_time'), float)
        all_reward_t.extend(rewards + t_start)
        all_threshold_t.extend(thresholds + t_start)
        all_lick_t.extend(licks + t_start)

        # quiescence: trial start -> 'go' event (response period start); response: 'go' event ->
        # threshold crossing (falls back to trial end if this trial never reached threshold)
        quiescence_end = t_start + go_times[0] if len(go_times) else None
        if quiescence_end is not None:
            response_end = t_start + thresholds[0] if len(thresholds) else t_end
        else:
            response_end = None
        trial_periods.append(dict(t_start=t_start, t_end=t_end,
                                   quiescence_end=quiescence_end, response_end=response_end))

        # the port stays closed (wherever the last reward left it) through the whole quiescence
        # period -- nothing touches it, and it isn't logged again until the 'go' event opens it
        # for the response period -- so hold the last known value (carried over from the previous
        # trial if this trial hasn't logged anything yet) until that first real sample, rather than
        # letting a plain connecting line ramp gradually across the silent gap
        lp_t, lp_pos = (experiment.TrialRewardPortPosition() & key & {'trial': trial}
                        ).fetch1('reward_port_position_time', 'reward_port_position_values')
        lp_t = np.asarray(lp_t, float)
        lp_pos = np.asarray(lp_pos, float)
        if len(lp_t):
            lead_val = lickport_carry if lickport_carry is not None else lp_pos[0]
            ext_t = np.concatenate([[0.0], lp_t, [t_end - t_start]])
            ext_pos = np.concatenate([[lead_val], lp_pos, [lp_pos[-1]]])
            lickport_carry = lp_pos[-1]
        elif lickport_carry is not None:
            ext_t = np.array([0.0, t_end - t_start])
            ext_pos = np.array([lickport_carry, lickport_carry])
        else:
            ext_t, ext_pos = np.array([]), np.array([])
        all_lickport_t.extend(ext_t + t_start)
        all_lickport_pos.extend(ext_pos)

    all_force_t       = np.array(all_force_t)
    all_force_0       = np.array(all_force_0)
    all_force_1       = np.array(all_force_1)
    all_reward_t      = np.array(all_reward_t)
    all_threshold_t   = np.array(all_threshold_t)
    all_lick_t        = np.array(all_lick_t)
    all_lickport_t    = np.array(all_lickport_t)
    all_lickport_pos  = np.array(all_lickport_pos)

    # --- force traces in the normalized LR/PA convention ---
    all_force_lr = lr_sign * (all_force_0 if lr_idx == 0 else all_force_1)
    all_force_pa = pa_sign * (all_force_1 if pa_idx == 1 else all_force_0)

    lp_shifted        = all_lickport_pos - np.min(all_lickport_pos)
    all_lickport_norm = lp_shifted / np.max(np.abs(lp_shifted))

    # --- video: use full VideoFileFrameTimes to support padding ---
    video_file_path, file_frame_times = _camera_video_file(videography, dj, key, camera_name, trials_needed)

    raw_t_end     = float((experiment.SessionTrial() & key & {'trial': int(trials_needed[-1])}).fetch1('trial_end_time'))
    t_video_start = max(float(trial_start_times[trials_needed[0]]) - pad_start, float(file_frame_times[0]))
    t_video_end   = min(raw_t_end + pad_end, float(file_frame_times[-1]))

    frame_abs_indices, frame_session_times = _camera_frame_window(file_frame_times, t_video_start, t_video_end)

    video_file_path_2, frame_abs_indices_2, frame_session_times_2 = None, None, None
    if camera_name_2:
        video_file_path_2, file_frame_times_2 = _camera_video_file(videography, dj, key, camera_name_2, trials_needed)
        frame_abs_indices_2, frame_session_times_2 = _camera_frame_window(file_frame_times_2, t_video_start, t_video_end)

    return dict(
        subject_id=subject_id, session=session, block=block,
        trial_start=trial_start, trial_end=trial_end,
        camera_name=camera_name, camera_name_2=camera_name_2,
        pad_start=pad_start, pad_end=pad_end,
        filter_method=filter_method, filter_window_ms=filter_window_ms,
        filter_sigma_ms=filter_sigma_ms, filter_polyorder=filter_polyorder,
        trials_needed=trials_needed,
        target_force_lut=target_force_lut, lut_extent=lut_extent,
        all_force_t=all_force_t, all_force_lr=all_force_lr, all_force_pa=all_force_pa,
        all_reward_t=all_reward_t, all_threshold_t=all_threshold_t, all_lick_t=all_lick_t,
        all_lickport_t=all_lickport_t, all_lickport_norm=all_lickport_norm,
        trial_periods=trial_periods,
        video_file_path=video_file_path,
        frame_abs_indices=frame_abs_indices, frame_session_times=frame_session_times,
        video_file_path_2=video_file_path_2,
        frame_abs_indices_2=frame_abs_indices_2, frame_session_times_2=frame_session_times_2,
        t_video_start=t_video_start, t_video_end=t_video_end,
    )


def _prepare_frame(bgr_frame, crop, clims, target_size=None):
    """target_size, if given, is (width, height) in pixels to downscale the cropped frame to
    before it's handed to imshow. Without this, imshow re-resamples the full camera-resolution
    array from scratch on every single draw() call (measured ~2.6x the cost of drawing an
    already-display-sized image), even though the displayed size never changes within a render -
    shrinking once here with cv2 (much faster than Agg's resampler) up front avoids paying that
    repeatedly. INTER_AREA is the appropriate/fast choice for shrinking (matches cv2 guidance for
    downscaling, avoiding the aliasing plain nearest/linear would introduce)."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    cl, cr, ct, cb = crop
    h, w = rgb.shape[:2]
    rgb = rgb[ct: h - cb if cb else None,
              cl: w - cr if cr else None]
    if target_size is not None and target_size != (rgb.shape[1], rgb.shape[0]):
        rgb = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)
    lo, hi = clims
    return np.clip((rgb.astype(float) - lo) / (hi - lo), 0, 1)


def _image_target_pixels(w_crop, h_crop, col_w_in, dpi):
    """Pixel size (width, height) the camera image column will actually be rendered at, given
    its allocated column width in inches (img_col_w) and the render dpi - used to downscale the
    raw camera frame to its true displayed size once per frame instead of leaving matplotlib to
    re-resample the full-resolution frame on every draw() call. See _prepare_frame."""
    target_w = max(1, int(round(col_w_in * dpi)))
    target_h = max(1, int(round(target_w * h_crop / w_crop)))
    return target_w, target_h


class _VideoFrameArtists:
    """Builds the video-overlay figure once (frame 0), then update() mutates only the artists
    that actually change per frame instead of rebuilding the whole Figure/Axes from scratch
    every time. Rebuilding was the actual rendering bottleneck (~700ms/frame, mostly matplotlib
    regenerating every tick label from nothing) even after fixing the separate video-seek cost -
    truly per-frame content is only the camera image(s) and the progressively-revealed force
    trajectory/feedback trace; everything else - target LUT panel above all, since it's the same
    for the whole block, but also every axis label/title/tick/limit - is drawn exactly once here
    and left alone, updated via cheap methods (.set_data()/.set_positions()/.remove()+re-add for
    the variable-count shaded spans) instead of matplotlib re-deriving it from nothing.

    update() is a straight port of what make_trial_video_frame_figure used to recompute inline
    on every call - same masks, same data, same draw order - just applied to existing artists
    instead of creating fresh ones; see that function (still the single-frame entry point, e.g.
    for preview_last_frame) for the one-frame equivalent this must stay behaviorally identical to.
    """

    def __init__(self, data, video_frame, t_now, crop, clims, tail_s=5, force_axis_limit=10.0,
                 lang='en', is_preview=False, video_frame_2=None, crop_2=None, clims_2=None,
                 lut_alpha=0.25, dpi=150):
        self.data = data
        self.crop = crop
        self.clims = clims
        self.crop_2 = crop_2 if crop_2 is not None else (0, 0, 0, 0)
        self.clims_2 = clims_2 if clims_2 is not None else clims
        self.tail_s = tail_s
        self.is_preview = is_preview
        self.labels = _LABELS[lang]
        has_cam2 = video_frame_2 is not None

        cl, cr, ct, cb = crop
        h, w = video_frame.shape[:2]
        h_crop = h - ct - (cb if cb else 0)
        w_crop = w - cl - (cr if cr else 0)
        # the camera image(s) sit in a fixed-width column so each image is always the same size
        # regardless of camera count; a second stacked image makes the figure taller, and the
        # other column's width is scaled up by that same factor so its own aspect ratio (and the
        # target/current force squares within it) doesn't get squeezed narrow by the extra height
        img_col_w = 7.5
        fig_h_1cam = max(img_col_w * h_crop / w_crop, 7)

        if has_cam2:
            h2, w2 = video_frame_2.shape[:2]
            cl2, cr2, ct2, cb2 = self.crop_2
            h_crop2 = h2 - ct2 - (cb2 if cb2 else 0)
            w_crop2 = w2 - cl2 - (cr2 if cr2 else 0)
            fig_h = max(img_col_w * h_crop / w_crop + img_col_w * h_crop2 / w_crop2, 7)
            other_col_w = img_col_w * (fig_h / fig_h_1cam)
        else:
            fig_h = fig_h_1cam
            other_col_w = img_col_w

        fig_w = other_col_w + img_col_w
        # dpi fixed at figure-creation time (matching what render_trial_video passes to
        # savefig(dpi=...) for this same figure) so _image_target_pixels below - which sizes the
        # pre-resized camera frame(s) - targets the same pixel dimensions the frame actually gets
        # rendered/saved at
        self.fig = plt.figure(figsize=[fig_w, fig_h], dpi=dpi)
        # camera frame(s) are downscaled to this once per update() instead of imshow silently
        # re-resampling the full camera-resolution array from scratch on every draw() - see
        # _prepare_frame/_image_target_pixels
        self.img_target_size = _image_target_pixels(w_crop, h_crop, img_col_w, dpi)
        self.img_target_size_2 = (_image_target_pixels(w_crop2, h_crop2, img_col_w, dpi)
                                   if has_cam2 else None)
        # camera image(s) on the right, everything else (feedback/force-vs-time/target/current)
        # on the left - left needs real margin now (unlike when this edge only ever held a
        # borderless axis('off') image), so tick/axis labels on the left column aren't clipped
        gs_outer = plt.GridSpec(1, 2, figure=self.fig, width_ratios=[other_col_w, img_col_w],
                                 left=0.05, right=0.99, top=0.97, bottom=0.06, wspace=0.3)
        if has_cam2:
            gs_img = gs_outer[1].subgridspec(2, 1, height_ratios=[h_crop / w_crop, h_crop2 / w_crop2], hspace=0.08)
            ax_img = self.fig.add_subplot(gs_img[0])
            ax_img_2 = self.fig.add_subplot(gs_img[1])
        else:
            ax_img = self.fig.add_subplot(gs_outer[1])
            ax_img_2 = None
        gs_other = gs_outer[0].subgridspec(2, 1, hspace=0.5)
        # feedback/lickport (top of the pair) shares its x axis with force-vs-time (bottom of the
        # pair), same as the block/session plots' equivalent stacked panels
        gs_top       = gs_other[0].subgridspec(2, 1, hspace=0.15)
        self.ax_force_time = self.fig.add_subplot(gs_top[1])
        self.ax_lickport   = self.fig.add_subplot(gs_top[0], sharex=self.ax_force_time)
        gs_bot       = gs_other[1].subgridspec(1, 2, wspace=0.1)
        ax_target    = self.fig.add_subplot(gs_bot[0])
        self.ax_current = self.fig.add_subplot(gs_bot[1], sharex=ax_target, sharey=ax_target)

        # --- static: camera image axes (title/off-axis set once; pixel content updates per frame) ---
        self.im1 = ax_img.imshow(_prepare_frame(video_frame, crop, clims, self.img_target_size))
        ax_img.axis('off')
        if is_preview:
            ax_img.set_title(f"{data['camera_name']}  subject={data['subject_id']}  session={data['session']}  "
                              f"block={data['block']}  trials {data['trial_start']}–{data['trial_end']}  [LAST FRAME]",
                              fontsize=11)
        else:
            ax_img.set_title(data['camera_name'], fontsize=9)
        self.im2 = None
        if has_cam2:
            self.im2 = ax_img_2.imshow(_prepare_frame(video_frame_2, self.crop_2, self.clims_2, self.img_target_size_2))
            ax_img_2.axis('off')
            ax_img_2.set_title(data['camera_name_2'], fontsize=9)

        # --- fully static: target LUT never changes within a block, so it's drawn once and
        # never touched again by update() - this is the panel that was previously getting
        # rebuilt from scratch on every single frame for no reason ---
        target_force_lut = data['target_force_lut']
        lut_extent = data['lut_extent']
        uniform_extent = [-force_axis_limit, force_axis_limit, -force_axis_limit, force_axis_limit]
        vmin, vmax = np.min(target_force_lut), np.max(target_force_lut)
        lut_disp = _pad_lut_to_range(target_force_lut, lut_extent, uniform_extent)
        ax_target.imshow(lut_disp, extent=uniform_extent, origin='lower',
                          cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        ax_target.set_xlim([-force_axis_limit, force_axis_limit])
        ax_target.set_ylim([-force_axis_limit, force_axis_limit])
        ax_target.set_xlabel(self.labels['force_x'], fontsize=10)
        ax_target.set_ylabel(self.labels['force_y'], fontsize=10)
        ax_target.set_title(self.labels['lbl_target'], fontsize=11)

        # --- fully static: same target LUT overlaid faintly on the Current panel, so the
        # trajectory can be read against the target without glancing over to the separate Target
        # panel; drawn before the trajectory artists below so it stays behind them, and kept
        # translucent (lut_alpha) so it doesn't drown out the black/gray trajectory on top of it ---
        self.ax_current.imshow(lut_disp, extent=uniform_extent, origin='lower',
                                cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax, alpha=lut_alpha)

        # --- dynamic: force trajectory (placeholder artists now, real data via update() below) ---
        self.line_all,  = self.ax_current.plot([], [], '-', color='black', linewidth=0.7,alpha = .5)
        self.line_tail, = self.ax_current.plot([], [], 'k-', linewidth=2)
        self.marker_now, = self.ax_current.plot([], [], 'ro', markersize=12)
        self.ax_current.set_xlabel(self.labels['force_x'], fontsize=10)
        self.ax_current.set_title(self.labels['lbl_current'], fontsize=11)
        plt.setp(self.ax_current.get_yticklabels(), visible=False)

        # --- static: feedback/lickport and force-vs-time axis limits/labels depend only on the
        # fixed video time window, not on t_now, so they're set once here ---
        t_video_start = data['t_video_start']
        t_video_end   = data['t_video_end']
        self.x0 = t_video_start  # reference for x-axis (0 = start of video window)
        self.ax_lickport.set_xlim([0, t_video_end - self.x0])
        # explicit ylim, not just autoscale: the line/eventplot/star artists below are all
        # created once here with empty placeholder data and filled in later via .set_data(),
        # which - unlike a fresh ax.plot() call - does NOT trigger matplotlib's autoscale, so
        # without this the axis stayed stuck at whatever range it drew for that empty data and
        # clipped off the "target" (y=1) end once the trace actually reached it. 0=start/1=target
        # is a fixed normalized range for the whole block anyway, so a static ylim is correct
        # here regardless - padded a bit above 1 for the eventplot ticks at lineoffset=1.1
        self.ax_lickport.set_ylim([-0.08, 1.25])
        self.ax_lickport.set_yticks([0, 1])
        self.ax_lickport.set_yticklabels([self.labels['fb_start'], self.labels['fb_target']])
        self.ax_lickport.set_ylabel(self.labels['feedback'])
        plt.setp(self.ax_lickport.get_xticklabels(), visible=False)  # x axis shared with the panel below

        self.ax_force_time.set_ylim([-force_axis_limit, force_axis_limit])
        self.ax_force_time.set_xlabel(self.labels['time'])
        self.ax_force_time.set_ylabel(self.labels['force_time'])
        self.ax_force_time.legend(
            handles=[plt.Line2D([0], [0], color='tab:blue', label=self.labels['lr_label']),
                     plt.Line2D([0], [0], color='tab:orange', label=self.labels['pa_label'])],
            loc='upper right', fontsize=7, ncol=2, framealpha=0.6)

        # --- dynamic: lickport/feedback trace (placeholder artists; spans/reward lines are
        # recreated each update() since their count and extents both change as more of the trial
        # is revealed) ---
        self.spans = []
        self.reward_lines = []
        self.lick_events = self.ax_lickport.eventplot(
            [], orientation='horizontal', lineoffsets=1.1, linelengths=0.15, colors='black')[0]
        self.line_lickport, = self.ax_lickport.step([], [], 'k-', where='post')
        self.stars, = self.ax_lickport.plot([], [], '*', color='gold', markersize=14,
                                             markeredgecolor='orange', markeredgewidth=1, zorder=5)
        self.vline = self.ax_lickport.axvline(t_now - self.x0, color='r', linestyle='--', linewidth=1)

        # --- dynamic: force-vs-time (same LR/PA traces as the Current panel, plotted over time
        # instead of space; matches the block/session plots' force-vs-time panel) ---
        self.line_force_lr, = self.ax_force_time.plot([], [], '-', color='tab:blue', linewidth=0.7, alpha=0.7)
        self.line_force_pa, = self.ax_force_time.plot([], [], '-', color='tab:orange', linewidth=0.7, alpha=0.7)
        self.vline2 = self.ax_force_time.axvline(t_now - self.x0, color='r', linestyle='--', linewidth=1)

        self.update(video_frame, t_now, video_frame_2=video_frame_2)

    def _shade_periods(self, ax, t_end):
        """Quiescence/response shading for one axis, clipped to what's visible so far (t_end) -
        matching behavior_plots' _shade_trial_periods. Appends the new patches to self.spans so
        update() can remove them next frame; called once per axis that needs this shading."""
        data = self.data
        t_video_start = data['t_video_start']
        x0 = self.x0
        for period in data['trial_periods']:
            p_start = period['t_start']
            if p_start > t_end or period['t_end'] < t_video_start:
                continue
            q_end = period['quiescence_end']
            if q_end is None:
                continue
            gray_start, gray_end = max(p_start, t_video_start), min(q_end, t_end)
            if gray_end > gray_start:
                self.spans.append(ax.axvspan(gray_start - x0, gray_end - x0, color='gray', alpha=0.15, linewidth=0))
            if t_end > q_end:
                r_end = period['response_end'] if period['response_end'] is not None else period['t_end']
                gold_end = min(r_end, t_end)
                if gold_end > q_end:
                    self.spans.append(ax.axvspan(q_end - x0, gold_end - x0, color='gold', alpha=0.15, linewidth=0))

    def update(self, video_frame, t_now, video_frame_2=None):
        data = self.data
        self.im1.set_data(_prepare_frame(video_frame, self.crop, self.clims, self.img_target_size))
        if self.im2 is not None and video_frame_2 is not None:
            self.im2.set_data(_prepare_frame(video_frame_2, self.crop_2, self.clims_2, self.img_target_size_2))

        all_force_t  = data['all_force_t']
        all_force_lr = data['all_force_lr']
        all_force_pa = data['all_force_pa']
        # the whole trajectory up to now, in light grey, for context -- drawn first (lowest
        # zorder-equivalent, i.e. created first) so the black tail_s-second tail draws on top
        # of it for the most recent segment
        mask_all = all_force_t <= t_now
        self.line_all.set_data(all_force_lr[mask_all], all_force_pa[mask_all])
        mask = (all_force_t >= t_now - self.tail_s) & (all_force_t <= t_now)
        self.line_tail.set_data(all_force_lr[mask], all_force_pa[mask])
        if mask.any():
            self.marker_now.set_data([all_force_lr[mask][-1]], [all_force_pa[mask][-1]])
        else:
            self.marker_now.set_data([], [])

        t_video_start = data['t_video_start']
        t_video_end   = data['t_video_end']
        t_end = t_video_end if self.is_preview else t_now
        x0 = self.x0

        all_lickport_t    = data['all_lickport_t']
        all_lickport_norm = data['all_lickport_norm']
        all_threshold_t   = data['all_threshold_t']
        all_lick_t        = data['all_lick_t']
        all_reward_t      = data['all_reward_t']

        # quiescence/response shading + reward markers, clipped to what's visible so far -- like
        # the trace/markers above, they only reveal up to t_end so the video doesn't spoil what
        # hasn't happened yet. Count and extent both change as t_end advances, so - unlike the
        # line/image artists above - these can't be updated in place; drop the old ones and add
        # fresh ones each frame instead, which is still far cheaper than rebuilding the figure.
        for span in self.spans:
            span.remove()
        self.spans = []
        self._shade_periods(self.ax_lickport, t_end)
        self._shade_periods(self.ax_force_time, t_end)

        for line in self.reward_lines:
            line.remove()
        self.reward_lines = []
        if all_reward_t.size:
            reward_mask = (all_reward_t >= t_video_start) & (all_reward_t <= t_end)
            for rt in all_reward_t[reward_mask]:
                self.reward_lines.append(
                    self.ax_force_time.axvline(rt - x0, color='green', linewidth=1.2, alpha=0.8))

        if all_lick_t.size:
            lick_mask = (all_lick_t >= t_video_start) & (all_lick_t <= t_end)
            self.lick_events.set_positions(all_lick_t[lick_mask] - x0 if lick_mask.any() else [])

        lp_mask = (all_lickport_t >= t_video_start) & (all_lickport_t <= t_end)
        # step, not a plain connecting line: load_trial_video_data already inserts held-value
        # points at each gap in the raw log (e.g. an unlogged quiescence period), so a straight
        # line would draw those as a gradual ramp instead of the flat hold they actually represent
        self.line_lickport.set_data(all_lickport_t[lp_mask] - x0, all_lickport_norm[lp_mask])
        if all_threshold_t.size:
            thr_mask = (all_threshold_t >= t_video_start) & (all_threshold_t <= t_end)
            self.stars.set_data(all_threshold_t[thr_mask] - x0, np.ones(thr_mask.sum()))
        self.vline.set_xdata([t_now - x0, t_now - x0])

        force_mask = (all_force_t >= t_video_start) & (all_force_t <= t_end)
        self.line_force_lr.set_data(all_force_t[force_mask] - x0, all_force_lr[force_mask])
        self.line_force_pa.set_data(all_force_t[force_mask] - x0, all_force_pa[force_mask])
        self.vline2.set_xdata([t_now - x0, t_now - x0])

    def close(self):
        plt.close(self.fig)


def make_trial_video_frame_figure(data, video_frame, t_now, crop, clims, tail_s=5,
                                   force_axis_limit=10.0, lang='en', is_preview=False,
                                   video_frame_2=None, crop_2=None, clims_2=None, lut_alpha=0.25,
                                   dpi=150):
    """Single video-overlay figure for one frame: raw frame(s), target LUT, current force, feedback trace.

    If video_frame_2 is given, two camera images are stacked in the left column (video_frame on
    top, video_frame_2 below, each cropped/brightness-scaled by its own crop/clims pair — crop_2
    falls back to crop, clims_2 falls back to clims, if not given) instead of the single image
    spanning the full column height; the right column (feedback trace, target/current force) is
    unchanged either way.

    lut_alpha controls the opacity of the target LUT echoed behind the trajectory in the Current
    panel (0 hides it entirely, 1 matches the Target panel's own opacity).

    A thin single-frame entry point (used by preview_last_frame) over _VideoFrameArtists, which
    render_trial_video drives directly across many frames instead, updating only what changes
    per frame rather than paying for this from-scratch build on every one of them.
    """
    return _VideoFrameArtists(data, video_frame, t_now, crop, clims, tail_s=tail_s,
                               force_axis_limit=force_axis_limit, lang=lang, is_preview=is_preview,
                               video_frame_2=video_frame_2, crop_2=crop_2, clims_2=clims_2,
                               lut_alpha=lut_alpha, dpi=dpi).fig


def _read_last_frame(video_file_path, frame_index):
    """Open video_file_path, seek to frame_index, and read it.

    Raises a clear, file-naming error instead of letting a failed seek/read (wrong or missing
    file, corrupt file, frame index past the end, ...) surface later as OpenCV's cryptic
    "cvtColor: !_src.empty()" assertion.
    """
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video file: {video_file_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_index} from video file: {video_file_path}")
    return frame


def preview_last_frame(data, crop, clim_pct=(0, 95), tail_s=5, force_axis_limit=10.0, lang='en',
                        crop_2=None, lut_alpha=0.50, dpi=150):
    """Preview figure for the last frame of the video window; also returns the raw frame(s) and
    clims (each camera's own brightness scaling, computed independently, since two different
    physical cameras rarely share the same brightness distribution) so the same scaling can be
    reused for the full render.

    If data has a second camera (camera_name_2), its last frame is fetched too and returned as
    last_frame_2 with its own clims_2 (both None otherwise); crop_2 only matters when a second
    camera is present.
    """
    last_frame = _read_last_frame(data['video_file_path'], data['frame_abs_indices'][-1])
    gray  = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
    clims = np.percentile(gray.flatten(), list(clim_pct))

    last_frame_2, clims_2 = None, None
    if data.get('camera_name_2'):
        last_frame_2 = _read_last_frame(data['video_file_path_2'], data['frame_abs_indices_2'][-1])
        gray_2 = cv2.cvtColor(last_frame_2, cv2.COLOR_BGR2GRAY)
        clims_2 = np.percentile(gray_2.flatten(), list(clim_pct))

    fig = make_trial_video_frame_figure(data, last_frame, data['frame_session_times'][-1],
                                         crop, clims, tail_s=tail_s, force_axis_limit=force_axis_limit,
                                         lang=lang, is_preview=True,
                                         video_frame_2=last_frame_2, crop_2=crop_2, clims_2=clims_2,
                                         lut_alpha=lut_alpha, dpi=dpi)
    return fig, last_frame, clims, last_frame_2, clims_2


def _advance_to_frame(cap, current_pos, target_pos, max_skip_reads=60):
    """Position `cap` so its next .read() returns frame `target_pos`, preferring plain
    sequential .read() calls over cv2.CAP_PROP_POS_FRAMES seeking whenever that's cheap.

    render_trial_video's frame indices only ever increase (built from a sorted time array via
    searchsorted), so consecutive targets are usually 1-3 frames apart - i.e. this is really
    sequential playback. But cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, ...) doesn't know that:
    on inter-frame-compressed video (h264/h265 in an mkv, as used here) it reseeks to the
    nearest preceding keyframe and redecodes forward every time, even for a 1-frame hop -
    measured ~80x slower here than just calling .read() again and discarding what you don't
    want. So: read-and-discard forward for a short hop, and only fall back to an actual seek
    for the first frame, a backward jump, or a gap wide enough that reading through it would
    cost more than a reseek would (empirically break-even is around 100 frames here; 60 stays
    safely under that while comfortably covering real gaps, which are usually 1-3 frames).

    current_pos: the frame index the next .read() currently returns, or None if unknown (forces
    a seek - e.g. after a previous read failed). Returns the position callers should record
    after their own .read() succeeds - always target_pos, since that's what .read() will yield.
    """
    if current_pos is not None and 0 <= target_pos - current_pos <= max_skip_reads:
        for _ in range(target_pos - current_pos):
            cap.read()
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_pos)
    return target_pos


def _count_pngs(tmp_dir):
    """Live count of rendered-frame PNGs already on disk - a directory listing is cheap enough
    (thousands of entries) to poll a couple times a second for progress reporting; see
    render_trial_video's progress_callback."""
    return sum(1 for f in os.listdir(tmp_dir) if f.endswith('.png'))


def _chunk_ranges(n, n_chunks):
    """Split range(n) into up to n_chunks contiguous, near-equal (start, end) slices - used to
    hand each render worker a contiguous run of output frames (so its own video seek stays the
    cheap sequential-read case _advance_to_frame is built for, rather than random-access)."""
    n_chunks = max(1, min(n_chunks, n)) if n else 1
    base, rem = divmod(n, n_chunks)
    starts = []
    pos = 0
    for i in range(n_chunks):
        size = base + (1 if i < rem else 0)
        if size:
            starts.append((pos, pos + size))
        pos += size
    return starts


def _render_frame_chunk(data, crop, clims, tail_s, force_axis_limit, lang, crop_2, clims_2,
                         lut_alpha, dpi, tmp_dir, video_fps, chunk_specs):
    """Render one contiguous run of output frames to numbered PNGs in tmp_dir.

    chunk_specs: list of (frame_i, abs_frame, t_now, abs_frame_2_or_None), frame_i being the
    frame's position in the *whole* render (not just this chunk) so its PNG filename slots into
    the single final frame_%05d.png sequence ffmpeg stitches, regardless of which worker made it.

    Runs in its own process (see render_trial_video, which dispatches these across a
    ProcessPoolExecutor): each worker builds its own _VideoFrameArtists/opens its own
    cv2.VideoCapture(s), since neither is safe to share across processes, and frames are fully
    independent of each other (update() recomputes everything from t_now and data, which is
    read-only) so this needs no coordination with other workers beyond disjoint frame_i ranges.
    """
    has_cam2 = bool(data.get('camera_name_2'))
    cap  = cv2.VideoCapture(data['video_file_path'])
    cap2 = cv2.VideoCapture(data['video_file_path_2']) if has_cam2 else None
    cam1_pos, cam2_pos = None, None
    artists = None
    n_written = 0
    try:
        for frame_i, abs_frame, t_now, abs_frame_2 in chunk_specs:
            _advance_to_frame(cap, cam1_pos, abs_frame)
            ret, frame = cap.read()
            cam1_pos = abs_frame + 1 if ret else None
            if not ret:
                break
            frame_2 = None
            if has_cam2 and abs_frame_2 is not None:
                _advance_to_frame(cap2, cam2_pos, abs_frame_2)
                ret2, frame_2 = cap2.read()
                cam2_pos = abs_frame_2 + 1 if ret2 else None
                if not ret2:
                    frame_2 = None
            if artists is None:
                artists = _VideoFrameArtists(data, frame, t_now, crop, clims, tail_s=tail_s,
                                              force_axis_limit=force_axis_limit, lang=lang,
                                              video_frame_2=frame_2, crop_2=crop_2, clims_2=clims_2,
                                              lut_alpha=lut_alpha, dpi=dpi)
            else:
                artists.update(frame, t_now, video_frame_2=frame_2)
            # compress_level=1 (default is much higher): these PNGs are throwaway - ffmpeg reads
            # them once below and then the whole tmp_dir is deleted - so there's no reason to
            # spend CPU on smaller file size. PNG compression is always lossless regardless of
            # level, so this is a pure speed win with zero effect on the rendered video's quality
            artists.fig.savefig(os.path.join(tmp_dir, f'frame_{frame_i:05d}.png'), dpi=dpi,
                                 pil_kwargs={'compress_level': 1})
            n_written += 1
    finally:
        cap.release()
        if cap2 is not None:
            cap2.release()
        if artists is not None:
            artists.close()
    return n_written


def render_trial_video(data, output_path, video_fps=20, playback_speed=1.0, crop=(0, 0, 0, 0),
                        clims=None, clim_pct=(0, 95), tail_s=5, force_axis_limit=10.0, lang='en',
                        crop_2=None, clims_2=None, lut_alpha=0.25, dpi=150, bitrate='3000k',
                        n_workers=None, progress_callback=None):
    """Render and stitch the full trial-range video with ffmpeg.

    Returns (output_path, clims, clims_2): the clims/clims_2 actually used, which is either
    what was passed in or - when not given - what got auto-computed from the last frame, so a
    caller that wants to record exactly what happened (see save_render_params) doesn't have to
    duplicate that auto-detection logic itself.

    If data has a second camera (camera_name_2), its frames are read in lockstep with the first
    camera's (each output frame's timestamp is independently matched to each camera's own frame
    times) and stacked below it in the video via make_trial_video_frame_figure. clims_2 scales its
    brightness independently of clims (auto-computed from its own last frame if not given), since
    two different physical cameras rarely share the same brightness distribution.

    Frame rendering is embarrassingly parallel - each output frame only depends on its own t_now
    and the read-only `data` arrays, never on the frame before it - so the n_out output frames are
    split into n_workers (default: os.cpu_count()) contiguous chunks and rendered to PNGs by a
    ProcessPoolExecutor (processes, not threads: matplotlib's Agg draw() is CPU-bound and holds
    the GIL, so threads wouldn't actually overlap); a single ffmpeg pass then stitches the
    complete numbered PNG sequence exactly as when this ran single-process, so multiple workers
    never re-encode/concatenate separately and can't introduce a seam at chunk boundaries.

    progress_callback, if given, is called periodically (roughly twice a second) as
    progress_callback(n_done, n_total) - n_done is a live count of the PNGs actually present in
    the render's temp dir, i.e. real progress across every worker combined, not just how many of
    the (far coarser) per-worker chunks have completed entirely. Guaranteed to be called once
    with n_done=0 right before rendering starts and once with n_done==n_total right after the
    last frame is written (before the ffmpeg stitch, which isn't tracked per-frame). If not
    given, the same (n_done, n_total) progression is printed to stdout instead.

    dpi/bitrate together set output quality; see VIDEO_QUALITY_PRESETS for the low/medium/high/
    ultra presets the GUIs offer (dpi is by far the bigger lever on render *time*, since it's the
    pixel count each worker rasterizes per frame - bitrate only costs ffmpeg's final encode pass).
    """
    if clims is None:
        last_frame = _read_last_frame(data['video_file_path'], data['frame_abs_indices'][-1])
        gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        clims = np.percentile(gray.flatten(), list(clim_pct))

    has_cam2 = bool(data.get('camera_name_2'))
    if has_cam2 and clims_2 is None:
        last_frame_2 = _read_last_frame(data['video_file_path_2'], data['frame_abs_indices_2'][-1])
        gray_2 = cv2.cvtColor(last_frame_2, cv2.COLOR_BGR2GRAY)
        clims_2 = np.percentile(gray_2.flatten(), list(clim_pct))

    frame_abs_indices   = data['frame_abs_indices']
    frame_session_times = data['frame_session_times']
    t_video_start = data['t_video_start']
    t_video_end   = data['t_video_end']

    dt_real      = playback_speed / video_fps
    target_times = np.arange(t_video_start, t_video_end, dt_real)
    idxs         = np.searchsorted(frame_session_times, target_times).clip(0, len(frame_session_times) - 1)
    selected_abs   = frame_abs_indices[idxs]
    selected_times = frame_session_times[idxs]
    n_out = len(selected_abs)
    print(f'Rendering {n_out} frames at {playback_speed}x speed ({dt_real * 1000:.0f} ms real time per output frame) ...')

    if has_cam2:
        idxs_2 = np.searchsorted(data['frame_session_times_2'], target_times).clip(
            0, len(data['frame_session_times_2']) - 1)
        selected_abs_2 = data['frame_abs_indices_2'][idxs_2]

    chunk_specs = [
        (frame_i, int(selected_abs[frame_i]), selected_times[frame_i],
         int(selected_abs_2[frame_i]) if has_cam2 else None)
        for frame_i in range(n_out)
    ]
    n_workers = max(1, n_workers or int(os.cpu_count()/2) or 1)
    tmp_dir = tempfile.mkdtemp()

    def report(n_done):
        if progress_callback is not None:
            progress_callback(n_done, n_out)
        else:
            print(f'  {n_done}/{n_out}')

    try:
        ranges = _chunk_ranges(n_out, n_workers)  # empty if n_out == 0 - no frames to render at all
        print(f'Rendering across {len(ranges)} worker process(es) ...')
        report(0)
        with ProcessPoolExecutor(max_workers=max(1, len(ranges))) as pool:
            futures = [
                pool.submit(_render_frame_chunk, data, crop, clims, tail_s, force_axis_limit,
                            lang, crop_2, clims_2, lut_alpha, dpi, tmp_dir, video_fps,
                            chunk_specs[start:end])
                for start, end in ranges
            ]
            # workers write PNGs straight into tmp_dir as they go, independently of each other -
            # polling that count (rather than waiting on whole chunks via as_completed) is what
            # turns "n_workers coarse jumps" into real, roughly-continuous progress; wait(...,
            # timeout=...) both paces this poll and blocks efficiently instead of a manual sleep
            not_done = set(futures)
            while not_done:
                _, not_done = futures_wait(not_done, timeout=0.5)
                report(min(_count_pngs(tmp_dir), n_out))
            for future in futures:
                future.result()  # re-raise any worker exception now that all have finished
            report(n_out)

        print('Stitching with ffmpeg ...')
        subprocess.run([
            'ffmpeg', '-y',
            '-framerate', str(video_fps),
            '-i', os.path.join(tmp_dir, 'frame_%05d.png'),
            # yuv420p (needed for broad player compatibility) requires even width/height, but the
            # frame PNGs' pixel size comes from figsize*dpi and isn't guaranteed to land on an
            # even number (more likely with a second stacked camera row) -- trim to even here
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-r', str(video_fps), '-pix_fmt', 'yuv420p', '-b:v', bitrate,
            output_path
        ], check=True)
        print(f'Done: {output_path}')
    finally:
        shutil.rmtree(tmp_dir)

    return output_path, clims, clims_2


def save_render_params(data, output_path, video_fps, playback_speed, crop, clims,
                        clim_pct=(0, 95), tail_s=5, force_axis_limit=10.0, lang='en',
                        crop_2=None, clims_2=None, lut_alpha=0.25, dpi=150, bitrate='3000k'):
    """Write a JSON sidecar next to output_path recording every parameter needed to exactly
    reproduce this render: which subject/session/block/trials/camera(s), the trace filter,
    crop, and brightness settings actually used.

    A sidecar file rather than embedding this in the video's own container metadata: metadata
    doesn't reliably survive re-encoding, trimming, or upload to most video-sharing tools,
    while a JSON file is trivially readable/greppable/diffable with no video-specific tooling,
    and loads straight back into load_trial_video_data/render_trial_video kwargs (see
    load_render_params/rerender_from_params) to actually redo the render, not just document it.
    """
    params = dict(
        generated_at=datetime.now(timezone.utc).isoformat(),
        subject_id=data['subject_id'], session=data['session'], block=data['block'],
        trial_start=data['trial_start'], trial_end=data['trial_end'],
        trials_needed=[int(t) for t in data['trials_needed']],
        camera_name=data['camera_name'], camera_name_2=data['camera_name_2'],
        pad_start=data['pad_start'], pad_end=data['pad_end'],
        filter_method=data['filter_method'], filter_window_ms=data['filter_window_ms'],
        filter_sigma_ms=data['filter_sigma_ms'], filter_polyorder=data['filter_polyorder'],
        crop=list(crop), crop_2=list(crop_2) if crop_2 is not None else None,
        clims=[float(c) for c in clims], clims_2=[float(c) for c in clims_2] if clims_2 is not None else None,
        clim_pct=list(clim_pct), tail_s=tail_s, force_axis_limit=force_axis_limit, lang=lang,
        lut_alpha=lut_alpha, dpi=dpi, bitrate=bitrate, video_fps=video_fps,
        playback_speed=playback_speed, output_path=str(output_path),
    )
    params_path = Path(output_path).with_suffix('.json')
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    return params_path


def load_render_params(params_path):
    """Load a sidecar written by save_render_params(). See rerender_from_params to actually
    redo the render from it rather than just inspecting the recorded parameters."""
    with open(params_path) as f:
        return json.load(f)


def rerender_from_params(params_path, output_path=None):
    """Recreate a video exactly as recorded by save_render_params().

    output_path defaults to the original path (so the render overwrites it in place); pass a
    different one to render a fresh copy alongside the original instead. Returns the new
    (output_path, clims, clims_2) from render_trial_video.
    """
    params = load_render_params(params_path)
    data = load_trial_video_data(
        subject_id=params['subject_id'], session=params['session'], block=params['block'],
        trial_start=params['trial_start'], trial_end=params['trial_end'],
        camera_name=params['camera_name'], camera_name_2=params['camera_name_2'],
        pad_start=params['pad_start'], pad_end=params['pad_end'],
        filter_method=params['filter_method'], filter_window_ms=params['filter_window_ms'],
        filter_sigma_ms=params['filter_sigma_ms'], filter_polyorder=params['filter_polyorder'])
    return render_trial_video(
        data, output_path or params['output_path'],
        video_fps=params['video_fps'], playback_speed=params['playback_speed'],
        crop=tuple(params['crop']), crop_2=tuple(params['crop_2']) if params['crop_2'] else None,
        clims=np.array(params['clims']), clims_2=np.array(params['clims_2']) if params['clims_2'] else None,
        clim_pct=tuple(params['clim_pct']), tail_s=params['tail_s'],
        force_axis_limit=params['force_axis_limit'], lang=params['lang'],
        # .get: older sidecars predate lut_alpha/dpi/bitrate, fall back to the same defaults
        # render_trial_video uses
        lut_alpha=params.get('lut_alpha', 0.25), dpi=params.get('dpi', 150),
        bitrate=params.get('bitrate', '3000k'))
