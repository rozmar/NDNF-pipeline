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


def load_trial_video_data(subject_id, session, block, trial_start, trial_end, camera_name,
                           pad_start=1.0, pad_end=1.0):
    """Fetch and normalize everything needed to render a trial-range video for one block.

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
    all_reward_t, all_threshold_t = [], []
    all_lickport_t, all_lickport_pos = [], []
    trial_start_times = {}

    for trial in trials_needed:
        t_start = float((experiment.SessionTrial() & key & {'trial': trial}).fetch1('trial_start_time'))
        trial_start_times[trial] = t_start

        ft = (experiment.TrialForceTrace()                & key & {'trial': trial}).fetch1('force_trace_time')
        f0 = (experiment.TrialForceTrace.TrialForceAxis() & key & {'trial': trial, 'force_axis_idx': 0}).fetch1('force_trace_value')
        f1 = (experiment.TrialForceTrace.TrialForceAxis() & key & {'trial': trial, 'force_axis_idx': 1}).fetch1('force_trace_value')
        all_force_t.extend(ft + t_start);  all_force_0.extend(f0);  all_force_1.extend(f1)

        rewards    = np.asarray((experiment.TrialEvent() & key & {'trial': trial, 'trial_event_type': 'reward'}).fetch('trial_event_time'), float)
        thresholds = np.asarray((experiment.TrialEvent() & key & {'trial': trial, 'trial_event_type': 'threshold crossing'}).fetch('trial_event_time'), float)
        all_reward_t.extend(rewards + t_start)
        all_threshold_t.extend(thresholds + t_start)

        lp_t, lp_pos = (experiment.TrialRewardPortPosition() & key & {'trial': trial}
                        ).fetch1('reward_port_position_time', 'reward_port_position_values')
        all_lickport_t.extend(lp_t + t_start)
        all_lickport_pos.extend(lp_pos)

    all_force_t       = np.array(all_force_t)
    all_force_0       = np.array(all_force_0)
    all_force_1       = np.array(all_force_1)
    all_reward_t      = np.array(all_reward_t)
    all_threshold_t   = np.array(all_threshold_t)
    all_lickport_t    = np.array(all_lickport_t)
    all_lickport_pos  = np.array(all_lickport_pos)

    # --- force traces in the normalized LR/PA convention ---
    all_force_lr = lr_sign * (all_force_0 if lr_idx == 0 else all_force_1)
    all_force_pa = pa_sign * (all_force_1 if pa_idx == 1 else all_force_0)

    lp_shifted        = all_lickport_pos - np.min(all_lickport_pos)
    all_lickport_norm = lp_shifted / np.max(np.abs(lp_shifted))

    # --- video: use full VideoFileFrameTimes to support padding ---
    current_file_idx = None
    video_file_path  = None
    for trial in trials_needed:
        tv = (videography.TrialVideo() & key & {'device': camera_name, 'trial': trial}).fetch1()
        if current_file_idx is None:
            current_file_idx = tv['video_file_idx']
            vf = (videography.VideoFile() & key & {'device': camera_name,
                                                   'video_file_idx': current_file_idx}).fetch1()
            video_file_path = os.path.join(dj.config['path.raw_data'], vf['file_path'])

    file_frame_times = (videography.VideoFileFrameTimes() & key & {
        'device': camera_name, 'video_file_idx': current_file_idx}).fetch1('frame_times')

    raw_t_end     = float((experiment.SessionTrial() & key & {'trial': int(trials_needed[-1])}).fetch1('trial_end_time'))
    t_video_start = max(float(trial_start_times[trials_needed[0]]) - pad_start, float(file_frame_times[0]))
    t_video_end   = min(raw_t_end + pad_end, float(file_frame_times[-1]))

    frame_mask          = (file_frame_times >= t_video_start) & (file_frame_times <= t_video_end)
    frame_abs_indices   = np.where(frame_mask)[0]        # absolute frame numbers for cv2
    frame_session_times = file_frame_times[frame_mask]    # session-relative timestamps

    return dict(
        subject_id=subject_id, session=session, block=block,
        trial_start=trial_start, trial_end=trial_end, camera_name=camera_name,
        trials_needed=trials_needed,
        target_force_lut=target_force_lut, lut_extent=lut_extent,
        all_force_t=all_force_t, all_force_lr=all_force_lr, all_force_pa=all_force_pa,
        all_reward_t=all_reward_t, all_threshold_t=all_threshold_t,
        all_lickport_t=all_lickport_t, all_lickport_norm=all_lickport_norm,
        video_file_path=video_file_path,
        frame_abs_indices=frame_abs_indices, frame_session_times=frame_session_times,
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
                                   force_axis_limit=10.0, lang='en', is_preview=False):
    """Single video-overlay figure for one frame: raw frame, target LUT, current force, feedback trace."""
    labels = _LABELS[lang]
    cl, cr, ct, cb = crop
    h, w = video_frame.shape[:2]
    h_crop = h - ct - (cb if cb else 0)
    w_crop = w - cl - (cr if cr else 0)
    fig_w = 15
    fig_h = max(fig_w / 2 * h_crop / w_crop, 7)

    fig = plt.figure(figsize=[fig_w, fig_h])
    gs = plt.GridSpec(2, 2, figure=fig, width_ratios=[1, 1],
                       left=0.01, right=0.99, top=0.97, bottom=0.06,
                       hspace=0.5, wspace=0.3)
    ax_img      = fig.add_subplot(gs[:, 0])
    ax_lickport = fig.add_subplot(gs[0, 1])
    gs_bot      = gs[1, 1].subgridspec(1, 2, wspace=0.1)
    ax_target   = fig.add_subplot(gs_bot[0])
    ax_current  = fig.add_subplot(gs_bot[1], sharex=ax_target, sharey=ax_target)

    ax_img.imshow(_prepare_frame(video_frame, crop, clims))
    ax_img.axis('off')
    if is_preview:
        ax_img.set_title(f"subject={data['subject_id']}  session={data['session']}  block={data['block']}  "
                          f"trials {data['trial_start']}–{data['trial_end']}  [LAST FRAME]", fontsize=11)

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

    lp_mask = (all_lickport_t >= t_video_start) & (all_lickport_t <= t_end)
    ax_lickport.plot(all_lickport_t[lp_mask] - x0, all_lickport_norm[lp_mask], 'k-')
    ax_lickport.plot([0, t_video_end - x0], [1, 1], 'r--', linewidth=1)
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
    ax_lickport.set_yticks([0, 1])
    ax_lickport.set_yticklabels([labels['fb_start'], labels['fb_target']])
    ax_lickport.set_xlabel(labels['time'])
    ax_lickport.set_ylabel(labels['feedback'])

    return fig


def preview_last_frame(data, crop, clim_pct=(0, 95), tail_s=5, force_axis_limit=10.0, lang='en'):
    """Preview figure for the last frame of the video window; also returns the raw frame and clims
    so the same brightness scaling can be reused for the full render."""
    cap = cv2.VideoCapture(data['video_file_path'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(data['frame_abs_indices'][-1]))
    ret, last_frame = cap.read()
    cap.release()

    gray  = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
    clims = np.percentile(gray.flatten(), list(clim_pct))

    fig = make_trial_video_frame_figure(data, last_frame, data['frame_session_times'][-1],
                                         crop, clims, tail_s=tail_s, force_axis_limit=force_axis_limit,
                                         lang=lang, is_preview=True)
    return fig, last_frame, clims


def render_trial_video(data, output_path, video_fps=20, playback_speed=1.0, crop=(0, 0, 0, 0),
                        clims=None, clim_pct=(0, 95), tail_s=5, force_axis_limit=10.0, lang='en'):
    """Render and stitch the full trial-range video with ffmpeg. Returns output_path."""
    if clims is None:
        cap = cv2.VideoCapture(data['video_file_path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(data['frame_abs_indices'][-1]))
        ret, last_frame = cap.read()
        cap.release()
        gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        clims = np.percentile(gray.flatten(), list(clim_pct))

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

    cap     = cv2.VideoCapture(data['video_file_path'])
    tmp_dir = tempfile.mkdtemp()
    try:
        for frame_i, (abs_frame, t_now) in enumerate(zip(selected_abs, selected_times)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(abs_frame))
            ret, frame = cap.read()
            if not ret:
                break
            fig = make_trial_video_frame_figure(data, frame, t_now, crop, clims,
                                                 tail_s=tail_s, force_axis_limit=force_axis_limit, lang=lang)
            plt.savefig(os.path.join(tmp_dir, f'frame_{frame_i:05d}.png'), dpi=150)
            plt.close(fig)
            if frame_i % 50 == 0:
                print(f'  {frame_i}/{n_out}')
        cap.release()
        print('Stitching with ffmpeg ...')
        subprocess.run([
            'ffmpeg', '-y',
            '-framerate', str(video_fps),
            '-i', os.path.join(tmp_dir, 'frame_%05d.png'),
            '-r', str(video_fps), '-pix_fmt', 'yuv420p', '-b:v', '3000k',
            output_path
        ], check=True)
        print(f'Done: {output_path}')
    finally:
        shutil.rmtree(tmp_dir)

    return output_path
