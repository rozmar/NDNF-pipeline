import os
import subprocess
import tempfile
import shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2
import datajoint as dj

_LABELS = {
    'en': dict(time='Time (s)', feedback='Feedback',
               fb_start='start', fb_target='target',
               force_x='Left - Right (g)',
               force_y='Posterior - Anterior (g)',
               lbl_target='Target', lbl_current='Current'),
    'hu': dict(time='Idő (mp)', feedback='Visszajelzés',
               fb_start='start', fb_target='cél',
               force_x='Bal - Jobb erő (g)',
               force_y='Hátsó - Elülső erő (g)',
               lbl_target='Cél', lbl_current='Aktuális'),
}


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
        video_file_path = os.path.join(dj.config['path.raw_data'], vf['file_path'])
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
                           camera_name_2=None, pad_start=1.0, pad_end=1.0):
    """Fetch and normalize everything needed to render a trial-range video for one block.

    If camera_name_2 is given, a second camera's frames are resolved too (clipped to the same
    [t_video_start, t_video_end] window as the first, so both stay in sync) for a side-by-side
    two-camera video; otherwise the video is single-camera as before.

    X is always Left-Right (L<0, R>0) and Y is always Posterior-Anterior (P<0, A>0),
    same convention as ndnf_pipeline.plot.behavior_plots.

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
    task_setting_id  = (experiment.Block() & key & {'block': block}).fetch1('task_setting_id')
    target_force_lut = (experiment.TaskSettings() & key & {'task_setting_id': task_setting_id}).fetch1('target_force_lut')

    force_axes = (experiment.TaskSettings.ForceAxis() & key & {'task_setting_id': task_setting_id}
                  ).fetch('force_axis_idx', 'target_force_axes', 'force_direction', order_by='force_axis_idx')
    force_axes_arrays = {idx: axes for idx, axes, _ in zip(*force_axes)}
    force_directions  = {idx: d    for idx, _, d  in zip(*force_axes)}

    # --- normalize to X=LR (L<0, R>0) and Y=PA (P<0, A>0) ---
    lr_idx = next(i for i in force_directions if force_directions[i] in ('LR', 'RL'))
    pa_idx = next(i for i in force_directions if force_directions[i] in ('PA', 'AP'))
    lr_sign = -1 if force_directions[lr_idx] == 'RL' else 1
    pa_sign = -1 if force_directions[pa_idx] == 'AP' else 1

    lr_ax = lr_sign * force_axes_arrays[lr_idx]
    pa_ax = pa_sign * force_axes_arrays[pa_idx]
    lr_extent = [float(lr_ax.min()), float(lr_ax.max())]
    pa_extent = [float(pa_ax.min()), float(pa_ax.max())]

    target_force_lut = target_force_lut.copy()
    if lr_sign == -1:
        target_force_lut = target_force_lut[:, ::-1] if lr_idx == 0 else target_force_lut[::-1, :]
    if pa_sign == -1:
        target_force_lut = target_force_lut[:, ::-1] if pa_idx == 0 else target_force_lut[::-1, :]
    if lr_idx == 1:
        target_force_lut = target_force_lut.T

    lut_extent = [lr_extent[0], lr_extent[1], pa_extent[0], pa_extent[1]]

    # --- per-trial data (force, feedback, rewards, threshold crossings) ---
    all_force_t, all_force_0, all_force_1 = [], [], []
    all_reward_t, all_threshold_t, all_lick_t = [], [], []
    all_lickport_t, all_lickport_pos = [], []
    trial_start_times = {}
    trial_periods = []  # per-trial (t_start, t_end, quiescence_end, response_end), all absolute

    lickport_carry = None  # last known lickport position, carried across the trial-to-trial gap
    for trial in trials_needed:
        t_start, t_end = (experiment.SessionTrial() & key & {'trial': trial}).fetch1('trial_start_time', 'trial_end_time')
        t_start, t_end = float(t_start), float(t_end)
        trial_start_times[trial] = t_start

        ft = (experiment.TrialForceTrace()                & key & {'trial': trial}).fetch1('force_trace_time')
        f0 = (experiment.TrialForceTrace.TrialForceAxis() & key & {'trial': trial, 'force_axis_idx': 0}).fetch1('force_trace_value')
        f1 = (experiment.TrialForceTrace.TrialForceAxis() & key & {'trial': trial, 'force_axis_idx': 1}).fetch1('force_trace_value')
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


def _prepare_frame(bgr_frame, crop, clims):
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    cl, cr, ct, cb = crop
    h, w = rgb.shape[:2]
    rgb = rgb[ct: h - cb if cb else None,
              cl: w - cr if cr else None]
    lo, hi = clims
    return np.clip((rgb.astype(float) - lo) / (hi - lo), 0, 1)


def make_trial_video_frame_figure(data, video_frame, t_now, crop, clims, tail_s=5,
                                   force_axis_limit=10.0, lang='en', is_preview=False,
                                   video_frame_2=None, crop_2=None, clims_2=None):
    """Single video-overlay figure for one frame: raw frame(s), target LUT, current force, feedback trace.

    If video_frame_2 is given, two camera images are stacked in the left column (video_frame on
    top, video_frame_2 below, each cropped/brightness-scaled by its own crop/clims pair — crop_2
    falls back to crop, clims_2 falls back to clims, if not given) instead of the single image
    spanning the full column height; the right column (feedback trace, target/current force) is
    unchanged either way.
    """
    labels = _LABELS[lang]
    cl, cr, ct, cb = crop
    h, w = video_frame.shape[:2]
    h_crop = h - ct - (cb if cb else 0)
    w_crop = w - cl - (cr if cr else 0)
    # the camera-image column keeps a fixed width regardless of camera count, so each camera
    # image is always the same size; a second stacked image makes the figure taller, and the
    # right column's width is scaled up by that same factor so its own aspect ratio (and the
    # target/current force squares within it) doesn't get squeezed narrow by the extra height
    left_col_w = 7.5
    fig_h_1cam = max(left_col_w * h_crop / w_crop, 7)

    if video_frame_2 is not None:
        crop_2 = crop_2 if crop_2 is not None else (0, 0, 0, 0)
        cl2, cr2, ct2, cb2 = crop_2
        h2, w2 = video_frame_2.shape[:2]
        h_crop2 = h2 - ct2 - (cb2 if cb2 else 0)
        w_crop2 = w2 - cl2 - (cr2 if cr2 else 0)
        fig_h = max(left_col_w * h_crop / w_crop + left_col_w * h_crop2 / w_crop2, 7)
        right_col_w = left_col_w * (fig_h / fig_h_1cam)
    else:
        fig_h = fig_h_1cam
        right_col_w = left_col_w

    fig_w = left_col_w + right_col_w
    fig = plt.figure(figsize=[fig_w, fig_h])
    gs_outer = plt.GridSpec(1, 2, figure=fig, width_ratios=[left_col_w, right_col_w],
                             left=0.01, right=0.99, top=0.97, bottom=0.06, wspace=0.3)
    if video_frame_2 is not None:
        gs_left = gs_outer[0].subgridspec(2, 1, height_ratios=[h_crop / w_crop, h_crop2 / w_crop2], hspace=0.08)
        ax_img = fig.add_subplot(gs_left[0])
        ax_img_2 = fig.add_subplot(gs_left[1])
    else:
        ax_img = fig.add_subplot(gs_outer[0])
        ax_img_2 = None
    gs_right    = gs_outer[1].subgridspec(2, 1, hspace=0.5)
    ax_lickport = fig.add_subplot(gs_right[0])
    gs_bot      = gs_right[1].subgridspec(1, 2, wspace=0.1)
    ax_target   = fig.add_subplot(gs_bot[0])
    ax_current  = fig.add_subplot(gs_bot[1], sharex=ax_target, sharey=ax_target)

    ax_img.imshow(_prepare_frame(video_frame, crop, clims))
    ax_img.axis('off')
    ax_img.set_title(data['camera_name'], fontsize=9)
    if is_preview:
        ax_img.set_title(f"{data['camera_name']}  subject={data['subject_id']}  session={data['session']}  "
                          f"block={data['block']}  trials {data['trial_start']}–{data['trial_end']}  [LAST FRAME]",
                          fontsize=11)
    if ax_img_2 is not None:
        ax_img_2.imshow(_prepare_frame(video_frame_2, crop_2, clims_2 if clims_2 is not None else clims))
        ax_img_2.axis('off')
        ax_img_2.set_title(data['camera_name_2'], fontsize=9)

    target_force_lut = data['target_force_lut']
    lut_extent = data['lut_extent']
    uniform_extent = [-force_axis_limit, force_axis_limit, -force_axis_limit, force_axis_limit]
    vmin, vmax = np.min(target_force_lut), np.max(target_force_lut)
    # fill the view outside the LUT's native extent with one of the LUT's own corner
    # values (its background level) instead of the global min, so a high corner value
    # doesn't render as a dark background patch
    background_value = target_force_lut[0, 0]
    ax_target.imshow(np.ones(target_force_lut.shape) * background_value,
                      extent=uniform_extent, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
    ax_target.imshow(target_force_lut, extent=lut_extent, origin='lower',
                      cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    ax_target.set_xlim([-force_axis_limit, force_axis_limit])
    ax_target.set_ylim([-force_axis_limit, force_axis_limit])
    ax_target.set_xlabel(labels['force_x'], fontsize=10)
    ax_target.set_ylabel(labels['force_y'], fontsize=10)
    ax_target.set_title(labels['lbl_target'], fontsize=11)

    all_force_t  = data['all_force_t']
    all_force_lr = data['all_force_lr']
    all_force_pa = data['all_force_pa']
    # the whole trajectory up to now, in light grey, for context -- drawn first so the black
    # tail_s-second tail (unchanged) draws on top of it for the most recent segment
    mask_all = all_force_t <= t_now
    ax_current.plot(all_force_lr[mask_all], all_force_pa[mask_all], '-', color='lightgray', linewidth=0.7)
    mask = (all_force_t >= t_now - tail_s) & (all_force_t <= t_now)
    ax_current.plot(all_force_lr[mask], all_force_pa[mask], 'k-', linewidth=2)
    if mask.any():
        ax_current.plot(all_force_lr[mask][-1], all_force_pa[mask][-1], 'ro', markersize=12)
    ax_current.set_xlabel(labels['force_x'], fontsize=10)
    ax_current.set_title(labels['lbl_current'], fontsize=11)
    plt.setp(ax_current.get_yticklabels(), visible=False)

    t_video_start = data['t_video_start']
    t_video_end   = data['t_video_end']
    t_end = t_video_end if is_preview else t_now
    x0    = t_video_start   # reference for x-axis (0 = start of video window)

    all_lickport_t    = data['all_lickport_t']
    all_lickport_norm = data['all_lickport_norm']
    all_reward_t      = data['all_reward_t']
    all_threshold_t   = data['all_threshold_t']
    all_lick_t        = data['all_lick_t']

    # quiescence/response shading per trial (matching the Block Detail tab's plot), clipped to
    # what's visible so far -- like the trace/markers below, it only reveals up to t_end so the
    # video doesn't spoil what hasn't happened yet
    for period in data['trial_periods']:
        p_start = period['t_start']
        if p_start > t_end or period['t_end'] < t_video_start:
            continue
        q_end = period['quiescence_end']
        if q_end is None:
            continue
        gray_start, gray_end = max(p_start, t_video_start), min(q_end, t_end)
        if gray_end > gray_start:
            ax_lickport.axvspan(gray_start - x0, gray_end - x0, color='gray', alpha=0.15, linewidth=0)
        if t_end > q_end:
            r_end = period['response_end'] if period['response_end'] is not None else period['t_end']
            gold_end = min(r_end, t_end)
            if gold_end > q_end:
                ax_lickport.axvspan(q_end - x0, gold_end - x0, color='gold', alpha=0.15, linewidth=0)

    if all_lick_t.size:
        lick_mask = (all_lick_t >= t_video_start) & (all_lick_t <= t_end)
        if lick_mask.any():
            ax_lickport.eventplot(all_lick_t[lick_mask] - x0, orientation='horizontal',
                                   lineoffsets=1.1, linelengths=0.15, colors='black')

    lp_mask = (all_lickport_t >= t_video_start) & (all_lickport_t <= t_end)
    # step, not a plain connecting line: load_trial_video_data already inserts held-value points
    # at each gap in the raw log (e.g. an unlogged quiescence period), so a straight line would
    # draw those as a gradual ramp instead of the flat hold they actually represent
    ax_lickport.step(all_lickport_t[lp_mask] - x0, all_lickport_norm[lp_mask], 'k-', where='post')
    if all_reward_t.size:
        rew_mask = (all_reward_t >= t_video_start) & (all_reward_t <= t_end)
        ax_lickport.plot(all_reward_t[rew_mask] - x0,
                          np.zeros(rew_mask.sum()) - 0.1, 'o', color='lightblue', markersize=10)
    if all_threshold_t.size:
        thr_mask = (all_threshold_t >= t_video_start) & (all_threshold_t <= t_end)
        ax_lickport.plot(all_threshold_t[thr_mask] - x0,
                          np.ones(thr_mask.sum()), '*', color='gold',
                          markersize=14, markeredgecolor='orange', markeredgewidth=1, zorder=5)
    ax_lickport.axvline(t_now - x0, color='r', linestyle='--', linewidth=1)
    ax_lickport.set_xlim([0, t_video_end - x0])
    ax_lickport.set_xticks([])
    ax_lickport.set_yticks([])

    return fig


def preview_last_frame(data, crop, clim_pct=(0, 95), tail_s=5, force_axis_limit=10.0, lang='en', crop_2=None):
    """Preview figure for the last frame of the video window; also returns the raw frame(s) and
    clims (each camera's own brightness scaling, computed independently, since two different
    physical cameras rarely share the same brightness distribution) so the same scaling can be
    reused for the full render.

    If data has a second camera (camera_name_2), its last frame is fetched too and returned as
    last_frame_2 with its own clims_2 (both None otherwise); crop_2 only matters when a second
    camera is present.
    """
    cap = cv2.VideoCapture(data['video_file_path'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(data['frame_abs_indices'][-1]))
    ret, last_frame = cap.read()
    cap.release()

    gray  = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
    clims = np.percentile(gray.flatten(), list(clim_pct))

    last_frame_2, clims_2 = None, None
    if data.get('camera_name_2'):
        cap2 = cv2.VideoCapture(data['video_file_path_2'])
        cap2.set(cv2.CAP_PROP_POS_FRAMES, int(data['frame_abs_indices_2'][-1]))
        ret2, last_frame_2 = cap2.read()
        cap2.release()
        gray_2 = cv2.cvtColor(last_frame_2, cv2.COLOR_BGR2GRAY)
        clims_2 = np.percentile(gray_2.flatten(), list(clim_pct))

    fig = make_trial_video_frame_figure(data, last_frame, data['frame_session_times'][-1],
                                         crop, clims, tail_s=tail_s, force_axis_limit=force_axis_limit,
                                         lang=lang, is_preview=True,
                                         video_frame_2=last_frame_2, crop_2=crop_2, clims_2=clims_2)
    return fig, last_frame, clims, last_frame_2, clims_2


def render_trial_video(data, output_path, video_fps=20, playback_speed=1.0, crop=(0, 0, 0, 0),
                        clims=None, clim_pct=(0, 95), tail_s=5, force_axis_limit=10.0, lang='en',
                        crop_2=None, clims_2=None):
    """Render and stitch the full trial-range video with ffmpeg. Returns output_path.

    If data has a second camera (camera_name_2), its frames are read in lockstep with the first
    camera's (each output frame's timestamp is independently matched to each camera's own frame
    times) and stacked below it in the video via make_trial_video_frame_figure. clims_2 scales its
    brightness independently of clims (auto-computed from its own last frame if not given), since
    two different physical cameras rarely share the same brightness distribution.
    """
    if clims is None:
        cap = cv2.VideoCapture(data['video_file_path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(data['frame_abs_indices'][-1]))
        ret, last_frame = cap.read()
        cap.release()
        gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        clims = np.percentile(gray.flatten(), list(clim_pct))

    has_cam2 = bool(data.get('camera_name_2'))
    if has_cam2 and clims_2 is None:
        cap2_probe = cv2.VideoCapture(data['video_file_path_2'])
        cap2_probe.set(cv2.CAP_PROP_POS_FRAMES, int(data['frame_abs_indices_2'][-1]))
        ret2, last_frame_2 = cap2_probe.read()
        cap2_probe.release()
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

    cap     = cv2.VideoCapture(data['video_file_path'])
    cap2    = cv2.VideoCapture(data['video_file_path_2']) if has_cam2 else None
    tmp_dir = tempfile.mkdtemp()
    try:
        for frame_i, (abs_frame, t_now) in enumerate(zip(selected_abs, selected_times)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(abs_frame))
            ret, frame = cap.read()
            if not ret:
                break
            frame_2 = None
            if has_cam2:
                cap2.set(cv2.CAP_PROP_POS_FRAMES, int(selected_abs_2[frame_i]))
                ret2, frame_2 = cap2.read()
                if not ret2:
                    frame_2 = None
            fig = make_trial_video_frame_figure(data, frame, t_now, crop, clims,
                                                 tail_s=tail_s, force_axis_limit=force_axis_limit, lang=lang,
                                                 video_frame_2=frame_2, crop_2=crop_2, clims_2=clims_2)
            plt.savefig(os.path.join(tmp_dir, f'frame_{frame_i:05d}.png'), dpi=150)
            plt.close(fig)
            if frame_i % 50 == 0:
                print(f'  {frame_i}/{n_out}')
        cap.release()
        if cap2 is not None:
            cap2.release()
        print('Stitching with ffmpeg ...')
        subprocess.run([
            'ffmpeg', '-y',
            '-framerate', str(video_fps),
            '-i', os.path.join(tmp_dir, 'frame_%05d.png'),
            # yuv420p (needed for broad player compatibility) requires even width/height, but the
            # frame PNGs' pixel size comes from figsize*dpi and isn't guaranteed to land on an
            # even number (more likely with a second stacked camera row) -- trim to even here
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-r', str(video_fps), '-pix_fmt', 'yuv420p', '-b:v', '3000k',
            output_path
        ], check=True)
        print(f'Done: {output_path}')
    finally:
        shutil.rmtree(tmp_dir)

    return output_path
