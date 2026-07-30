import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import numpy as np

DEFAULT_UNIFORM_FORCE_RANGE = np.asarray([-1, 1, -1, 1]) * 10


def _get_block_force_axes(experiment, subject_id, session, task_setting_id):
    # figure out which raw loadcell axis/direction is LR and which is PA
    force_axes = (experiment.TaskSettings.ForceAxis() & {'subject_id': subject_id, 'session': session, 'task_setting_id': task_setting_id}).fetch(as_dict=True)
    force_axes_dict = {}
    for fa in force_axes:
        force_axes_dict[fa['force_axis_idx']] = {'force_direction': fa['force_direction'],
                                                   'target_force_axes': fa['target_force_axes']}
    lr_idx = next(i for i in force_axes_dict if force_axes_dict[i]['force_direction'] in ('LR', 'RL'))
    pa_idx = next(i for i in force_axes_dict if force_axes_dict[i]['force_direction'] in ('PA', 'AP'))
    lr_sign = -1 if force_axes_dict[lr_idx]['force_direction'] == 'RL' else 1
    pa_sign = -1 if force_axes_dict[pa_idx]['force_direction'] == 'AP' else 1
    return force_axes_dict, lr_idx, pa_idx, lr_sign, pa_sign


def _normalize_lut(target_force_lut, force_axes_dict, lr_idx, pa_idx, lr_sign, pa_sign):
    # normalize a raw target_force_lut to X=Left-Right (L<0, R>0), Y=Posterior-Anterior (P<0, A>0)
    lr_ax = lr_sign * force_axes_dict[lr_idx]['target_force_axes']
    pa_ax = pa_sign * force_axes_dict[pa_idx]['target_force_axes']
    lr_extent = [float(lr_ax.min()), float(lr_ax.max())]
    pa_extent = [float(pa_ax.min()), float(pa_ax.max())]

    lut = target_force_lut.copy()
    if lr_sign == -1:
        lut = lut[::-1, :] if lr_idx == 0 else lut[:, ::-1]
    if pa_sign == -1:
        lut = lut[::-1, :] if pa_idx == 0 else lut[:, ::-1]
    if lr_idx == 1:
        lut = lut.T

    lut_extent = [lr_extent[0], lr_extent[1], pa_extent[0], pa_extent[1]]
    return lut, lut_extent, lr_extent, pa_extent


def _lr_pa_traces(force_traces_0, force_traces_1, lr_idx, pa_idx, lr_sign, pa_sign):
    lr_traces = [lr_sign * f for f in (force_traces_0 if lr_idx == 0 else force_traces_1)]
    pa_traces = [pa_sign * f for f in (force_traces_1 if pa_idx == 1 else force_traces_0)]
    return lr_traces, pa_traces


def _force_histogram(lr_traces, pa_traces, histrange, bins=50):
    # log-density 2D histogram of force, with -inf (empty bins) clipped to the finite minimum
    try:
        forcehist, binx, biny = np.histogram2d(
            np.concatenate(lr_traces), np.concatenate(pa_traces),
            range=histrange, bins=bins)
        forcehist = np.log(forcehist / forcehist.sum())
        forcehist_ = forcehist.copy()
        forcehist_[np.isinf(forcehist)] = 0
        forcehist[np.isinf(forcehist)] = np.nanmin(forcehist_.flatten())
        return forcehist, binx, biny
    except Exception:
        return None, None, None


def _pad_lut_to_range(lut, lut_extent, target_range):
    """Edge-pad lut.T to cover target_range by replicating border pixels outward.

    lut is [LR, PA] (dim0=LR, dim1=PA); returns a (PA+pad, LR+pad) array ready
    for imshow with origin='lower' and extent=target_range.
    """
    lr_min, lr_max, pa_min, pa_max = lut_extent
    u_lr_min, u_lr_max, u_pa_min, u_pa_max = target_range
    n_lr, n_pa = lut.shape
    dlr = (lr_max - lr_min) / max(n_lr - 1, 1)
    dpa = (pa_max - pa_min) / max(n_pa - 1, 1)
    pad_left  = max(0, int(round((lr_min  - u_lr_min) / dlr)))
    pad_right = max(0, int(round((u_lr_max - lr_max)  / dlr)))
    pad_bot   = max(0, int(round((pa_min  - u_pa_min) / dpa)))
    pad_top   = max(0, int(round((u_pa_max - pa_max)  / dpa)))
    return np.pad(lut.T, ((pad_bot, pad_top), (pad_left, pad_right)), mode='edge')


def _plot_force_trajectories(ax, lr_traces, pa_traces, force_uniform_range=True, uniform_force_range=None,
                              title='Force trajectories (early=blue, late=red)', ylabel=True, add_colorbar=True):
    # trajectories color-coded by trial (early=blue, late=red); shared by the blockwise
    # and sessionwise plots
    if uniform_force_range is None:
        uniform_force_range = DEFAULT_UNIFORM_FORCE_RANGE
    n_trials = len(lr_traces)
    cmap_traj = cm.coolwarm
    for i, (lr, pa) in enumerate(zip(lr_traces, pa_traces)):
        color = cmap_traj(i / max(n_trials - 1, 1))
        ax.plot(lr, pa, '-', color=color, alpha=0.5, linewidth=0.6)
    if force_uniform_range:
        ax.set_xlim(uniform_force_range[:2])
        ax.set_ylim(uniform_force_range[2:])
    ax.set_xlabel('Left - Right (g)')
    if ylabel:
        ax.set_ylabel('Posterior - Anterior (g)')
    if title:
        ax.set_title(title)
    if add_colorbar:
        sm = cm.ScalarMappable(cmap=cmap_traj, norm=plt.Normalize(vmin=1, vmax=max(n_trials, 1)))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='trial #')
    return ax


def plot_block_force_figure(subject_id, session, block, subtract_force_median=True,
                             force_uniform_range=True, uniform_force_range=None, fig_dir=None):
    """4-panel figure for one block: target LUT, performance, force distribution, force trajectories.

    X is always Left-Right (L<0, R>0) and Y is always Posterior-Anterior (P<0, A>0),
    regardless of which raw loadcell axis/direction the rig recorded.

    If fig_dir is given, saves to '{fig_dir}/{subject_id}_s{session}_b{block}.png'.
    Returns the figure.
    """
    # imported lazily: importing experiment.py at module load time would force a live
    # DataJoint connection as soon as `ndnf_pipeline.plot` is imported
    from ndnf_pipeline import experiment

    if uniform_force_range is None:
        uniform_force_range = DEFAULT_UNIFORM_FORCE_RANGE

    key = {'subject_id': subject_id, 'session': session, 'block': block}
    feedback_type = (experiment.Block() & key).fetch1('feedback_type')
    session_date, session_time = (experiment.Session() & {'subject_id': subject_id, 'session': session}).fetch1('session_date', 'session_time')
    task_setting_id, target_force_lut = (experiment.TaskSettings() * experiment.Block() & key).fetch1('task_setting_id', 'target_force_lut')
    force_axes_dict, lr_idx, pa_idx, lr_sign, pa_sign = _get_block_force_axes(experiment, subject_id, session, task_setting_id)
    lut, lut_extent, lr_extent, pa_extent = _normalize_lut(target_force_lut, force_axes_dict, lr_idx, pa_idx, lr_sign, pa_sign)

    fig = plt.figure(figsize=(12, 12))
    ax_target_LUT = fig.add_subplot(2, 2, 1)

    lut_vmin, lut_vmax = np.min(lut), np.max(lut)
    if force_uniform_range:
        lut_disp = _pad_lut_to_range(lut, lut_extent, uniform_force_range)
        disp_extent = uniform_force_range
    else:
        lut_disp = lut.T
        disp_extent = lut_extent
    ax_target_LUT.imshow(lut_disp, extent=disp_extent, origin='lower',
                          cmap='viridis', aspect='auto', vmin=lut_vmin, vmax=lut_vmax)
    ax_target_LUT.set_xlabel('Left - Right (g)')
    ax_target_LUT.set_ylabel('Posterior - Anterior (g)')
    if force_uniform_range:
        ax_target_LUT.set_xlim(uniform_force_range[:2])
        ax_target_LUT.set_ylim(uniform_force_range[2:])
    plt.colorbar(ax_target_LUT.images[-1], ax=ax_target_LUT, label='reward port speed')
    ax_target_LUT.set_title(f'{subject_id} s{session} b{block}: {feedback_type}')

    ax_performance = fig.add_subplot(2, 2, 3)
    rewarded_trial, time_to_reward = (experiment.TrialEvent() * experiment.BehaviorTrial() * experiment.Block() & {**key, 'trial_event_type': 'threshold crossing'}).fetch('trial', 'trial_event_time')
    ax_performance.set_xlabel('trial#')
    ax_performance.plot(rewarded_trial, time_to_reward, 'g.')
    if len(rewarded_trial) > 10:
        ax_performance.plot(np.convolve(rewarded_trial, np.ones(10) / 10, mode='valid'),
                             np.convolve(np.asarray(time_to_reward, float), np.ones(10) / 10, mode='valid'), 'g-')
    ax_performance.set_ylabel('time to get reward')

    ax_force_hist = fig.add_subplot(2, 2, 2)
    force_traces_0_ = (experiment.TrialForceTrace.TrialForceAxis() * experiment.BehaviorTrial() * experiment.Block() & {**key, 'force_axis_idx': 0}).fetch('force_trace_value')
    force_traces_1_ = (experiment.TrialForceTrace.TrialForceAxis() * experiment.BehaviorTrial() * experiment.Block() & {**key, 'force_axis_idx': 1}).fetch('force_trace_value')
    if subtract_force_median:
        f0_baseline = np.median(np.concatenate(force_traces_0_))
        f1_baseline = np.median(np.concatenate(force_traces_1_))
        force_traces_0 = [f - f0_baseline for f in force_traces_0_]
        force_traces_1 = [f - f1_baseline for f in force_traces_1_]
    else:
        force_traces_0 = list(force_traces_0_)
        force_traces_1 = list(force_traces_1_)

    lr_traces, pa_traces = _lr_pa_traces(force_traces_0, force_traces_1, lr_idx, pa_idx, lr_sign, pa_sign)

    histrange = [uniform_force_range[:2], uniform_force_range[2:]] if force_uniform_range else [lr_extent, pa_extent]
    forcehist, binx, biny = _force_histogram(lr_traces, pa_traces, histrange)
    if forcehist is not None:
        im_hist = ax_force_hist.imshow(forcehist.T, origin='lower',
                                        extent=[binx[0], binx[-1], biny[0], biny[-1]],
                                        aspect='auto', alpha=1)
        ax_force_hist.set_xlabel('Left - Right (g)')
        ax_force_hist.set_ylabel('Posterior - Anterior (g)')
        plt.colorbar(im_hist, ax=ax_force_hist, label='fraction of time spent')
    ax_force_hist.set_title(f'{session_date} {session_time}')

    ax_traj = fig.add_subplot(2, 2, 4)
    _plot_force_trajectories(ax_traj, lr_traces, pa_traces, force_uniform_range, uniform_force_range)

    if fig_dir is not None:
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(os.path.join(fig_dir, f'{subject_id}_s{session}_b{block}.png'), dpi=150)

    return fig


def plot_session_blocks_overview(subject_id, session, subtract_force_median=True,
                                  force_uniform_range=True, uniform_force_range=None, fig_dir=None):
    """Session-level overview across all of a session's blocks.

    Row 1: time-to-reward for every trial in the session (one panel), with each
           block's trial range highlighted and color-coded. Also marks, per trial,
           the time from threshold crossing to that trial's own end-of-trial marker
           ('x'); note this is NOT a true inter-trial gap (see the comment where it's
           computed below) — it's mostly a fixed hardware/task padding, except on
           trials bordering a real pause in the session, where it spikes.
    Row 2: each block's target LUT, side by side, sharing the same axes and colormap.
    Row 3: each block's force distribution histogram, side by side, sharing the
           same axes and colormap.
    Row 4: each block's force trajectories (early=blue, late=red), side by side.

    X is always Left-Right (L<0, R>0) and Y is always Posterior-Anterior (P<0, A>0).

    If fig_dir is given, saves to '{fig_dir}/{subject_id}_s{session}_blocks_overview.png'.
    Returns the figure.
    """
    from ndnf_pipeline import experiment

    if uniform_force_range is None:
        uniform_force_range = DEFAULT_UNIFORM_FORCE_RANGE

    available_blocks = np.sort((experiment.Block() & {'subject_id': subject_id, 'session': session}).fetch('block'))
    n_blocks = len(available_blocks)
    session_date, session_time = (experiment.Session() & {'subject_id': subject_id, 'session': session}).fetch1('session_date', 'session_time')
    block_colors = plt.cm.tab10(np.arange(max(n_blocks, 1)) % 10)

    # session-wide trial timing, used below to relate each rewarded trial's threshold-crossing
    # time to its own end-of-trial marker
    all_trials, all_trial_starts, all_trial_ends = (experiment.SessionTrial()
        & {'subject_id': subject_id, 'session': session}).fetch('trial', 'trial_start_time', 'trial_end_time')
    trial_start_by_num = dict(zip(all_trials.tolist(), np.asarray(all_trial_starts, float)))
    trial_end_by_num = dict(zip(all_trials.tolist(), np.asarray(all_trial_ends, float)))

    # --- gather everything needed per block up front, so rows 2 and 3 can share vmin/vmax ---
    block_data = []
    for block in available_blocks:
        key = {'subject_id': subject_id, 'session': session, 'block': block}
        task_setting_id, target_force_lut = (experiment.TaskSettings() * experiment.Block() & key).fetch1('task_setting_id', 'target_force_lut')
        force_axes_dict, lr_idx, pa_idx, lr_sign, pa_sign = _get_block_force_axes(experiment, subject_id, session, task_setting_id)
        lut, lut_extent, lr_extent, pa_extent = _normalize_lut(target_force_lut, force_axes_dict, lr_idx, pa_idx, lr_sign, pa_sign)

        force_traces_0_ = (experiment.TrialForceTrace.TrialForceAxis() * experiment.BehaviorTrial() * experiment.Block() & {**key, 'force_axis_idx': 0}).fetch('force_trace_value')
        force_traces_1_ = (experiment.TrialForceTrace.TrialForceAxis() * experiment.BehaviorTrial() * experiment.Block() & {**key, 'force_axis_idx': 1}).fetch('force_trace_value')
        if subtract_force_median:
            f0_baseline = np.median(np.concatenate(force_traces_0_))
            f1_baseline = np.median(np.concatenate(force_traces_1_))
            force_traces_0 = [f - f0_baseline for f in force_traces_0_]
            force_traces_1 = [f - f1_baseline for f in force_traces_1_]
        else:
            force_traces_0 = list(force_traces_0_)
            force_traces_1 = list(force_traces_1_)
        lr_traces, pa_traces = _lr_pa_traces(force_traces_0, force_traces_1, lr_idx, pa_idx, lr_sign, pa_sign)

        histrange = [uniform_force_range[:2], uniform_force_range[2:]] if force_uniform_range else [lr_extent, pa_extent]
        forcehist, binx, biny = _force_histogram(lr_traces, pa_traces, histrange)

        rewarded_trial, time_to_reward = (experiment.TrialEvent() * experiment.BehaviorTrial() * experiment.Block() & {**key, 'trial_event_type': 'threshold crossing'}).fetch('trial', 'trial_event_time')
        block_trials = (experiment.BehaviorTrial() & key).fetch('trial')

        # Time from threshold crossing to this trial's own end-of-trial marker. NOTE: at
        # ingestion (ingest_behavior.py), a trial's end marker is stamped at the *next*
        # trial's start timestamp, so on most trials this is really just a fixed
        # hardware/task padding after reward, not a true behavioral gap between trials —
        # it only spikes on trials that border a genuine pause/gap in the recording.
        post_reward_trial, post_reward_value = [], []
        for trial, t2r in zip(rewarded_trial, np.asarray(time_to_reward, float)):
            trial_end = trial_end_by_num.get(int(trial))
            if trial_end is None:
                continue
            post_reward_trial.append(trial)
            post_reward_value.append(trial_end - (trial_start_by_num[int(trial)] + t2r))

        block_data.append(dict(block=block, lut=lut, lut_extent=lut_extent,
                                forcehist=forcehist, binx=binx, biny=biny,
                                lr_traces=lr_traces, pa_traces=pa_traces,
                                rewarded_trial=rewarded_trial, time_to_reward=time_to_reward,
                                block_trials=block_trials,
                                post_reward_trial=np.array(post_reward_trial),
                                post_reward_value=np.array(post_reward_value)))

    lut_vmin = min(np.min(bd['lut']) for bd in block_data)
    lut_vmax = max(np.max(bd['lut']) for bd in block_data)
    hist_vals = [bd['forcehist'] for bd in block_data if bd['forcehist'] is not None]
    hist_vmin = min(np.min(h) for h in hist_vals) if hist_vals else None
    hist_vmax = max(np.max(h) for h in hist_vals) if hist_vals else None

    fig = plt.figure(figsize=(4 * max(n_blocks, 1), 13))
    gs = plt.GridSpec(4, max(n_blocks, 1), figure=fig, height_ratios=[1, 1.1, 1.1, 1.1], hspace=0.5, wspace=0.3)

    # --- row 1: time to reward across the whole session, blocks highlighted ---
    ax_perf = fig.add_subplot(gs[0, :])
    for bd, color in zip(block_data, block_colors):
        if len(bd['block_trials']):
            ax_perf.axvspan(bd['block_trials'].min() - 0.5, bd['block_trials'].max() + 0.5, color=color, alpha=0.15)
        ax_perf.plot(bd['rewarded_trial'], bd['time_to_reward'], '.', color=color, label=f"block {bd['block']}")
        if len(bd['rewarded_trial']) > 10:
            ax_perf.plot(np.convolve(bd['rewarded_trial'], np.ones(10) / 10, mode='valid'),
                         np.convolve(np.asarray(bd['time_to_reward'], float), np.ones(10) / 10, mode='valid'),
                         '-', color=color)
        if len(bd['post_reward_trial']):
            ax_perf.plot(bd['post_reward_trial'], bd['post_reward_value'], 'x', color=color, alpha=0.7)
    ax_perf.set_xlabel('trial#')
    ax_perf.set_ylabel('time (s)')
    ax_perf.set_title(f'{subject_id} s{session}  {session_date} {session_time}')
    block_legend = ax_perf.legend(loc='upper right', fontsize=8, ncol=min(max(n_blocks, 1), 6))
    ax_perf.add_artist(block_legend)
    marker_legend_handles = [
        Line2D([0], [0], marker='.', linestyle='None', color='black', label='time to reward'),
        Line2D([0], [0], marker='x', linestyle='None', color='black',
               label='threshold crossing -> trial-end marker (mostly fixed padding, not a true gap)'),
    ]
    ax_perf.legend(handles=marker_legend_handles, loc='upper left', fontsize=7)

    # --- row 2: target LUTs side by side, shared axes + colormap ---
    axes_lut = []
    im_lut = None
    for bi, (bd, color) in enumerate(zip(block_data, block_colors)):
        ax_lut = fig.add_subplot(gs[1, bi])
        if force_uniform_range:
            lut_disp = _pad_lut_to_range(bd['lut'], bd['lut_extent'], uniform_force_range)
            lut_disp_extent = uniform_force_range
        else:
            lut_disp = bd['lut'].T
            lut_disp_extent = bd['lut_extent']
        im_lut = ax_lut.imshow(lut_disp, extent=lut_disp_extent, origin='lower', cmap='viridis',
                                aspect='auto', vmin=lut_vmin, vmax=lut_vmax)
        if force_uniform_range:
            ax_lut.set_xlim(uniform_force_range[:2])
            ax_lut.set_ylim(uniform_force_range[2:])
        ax_lut.set_xlabel('Left - Right (g)')
        if bi == 0:
            ax_lut.set_ylabel('Posterior - Anterior (g)')
        ax_lut.set_title(f"block {bd['block']}", color=color)
        axes_lut.append(ax_lut)
    if im_lut is not None:
        fig.colorbar(im_lut, ax=axes_lut, label='reward port speed')

    # --- row 3: force histograms side by side, shared axes + colormap ---
    axes_hist = []
    im_hist = None
    for bi, (bd, color) in enumerate(zip(block_data, block_colors)):
        ax_hist = fig.add_subplot(gs[2, bi])
        if bd['forcehist'] is not None:
            im_hist = ax_hist.imshow(bd['forcehist'].T, origin='lower',
                                      extent=[bd['binx'][0], bd['binx'][-1], bd['biny'][0], bd['biny'][-1]],
                                      aspect='auto', vmin=hist_vmin, vmax=hist_vmax)
        elif force_uniform_range:
            ax_hist.set_xlim(uniform_force_range[:2])
            ax_hist.set_ylim(uniform_force_range[2:])
        ax_hist.set_xlabel('Left - Right (g)')
        if bi == 0:
            ax_hist.set_ylabel('Posterior - Anterior (g)')
        ax_hist.set_title(f"block {bd['block']}", color=color)
        axes_hist.append(ax_hist)
    if im_hist is not None:
        fig.colorbar(im_hist, ax=axes_hist, label='fraction of time spent (log)')

    # --- row 4: force trajectories side by side ---
    # one shared colorbar for the whole row: each block's trajectories are already
    # colored by *fractional* trial position (0=first, 1=last), so a single colorbar
    # labeled first...last applies across blocks even though they have different trial counts
    axes_traj = []
    for bi, (bd, color) in enumerate(zip(block_data, block_colors)):
        ax_traj = fig.add_subplot(gs[3, bi])
        _plot_force_trajectories(ax_traj, bd['lr_traces'], bd['pa_traces'], force_uniform_range, uniform_force_range,
                                  title=None, ylabel=(bi == 0), add_colorbar=False)
        ax_traj.set_title(f"block {bd['block']}", color=color)
        axes_traj.append(ax_traj)
    sm_traj = cm.ScalarMappable(cmap=cm.coolwarm, norm=plt.Normalize(vmin=0, vmax=1))
    sm_traj.set_array([])
    cbar_traj = fig.colorbar(sm_traj, ax=axes_traj, label='trial (within block)')
    cbar_traj.set_ticks([0, 1])
    cbar_traj.set_ticklabels(['first', 'last'])

    if fig_dir is not None:
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(os.path.join(fig_dir, f'{subject_id}_s{session}_blocks_overview.png'), dpi=150)

    return fig


def plot_water_restriction_overview(lab, highlight_subject_id=None, fig_dir=None):
    """Relative weight, water consumed, and raw weight over time for every subject on water restriction.

    If highlight_subject_id is given, that subject's lines are drawn thicker.
    If fig_dir is given, saves to '{fig_dir}/water_restriction_overview.png'.
    Returns the figure.
    """
    import matplotlib.dates as mdates

    mice_on_wr = lab.WaterRestriction().fetch('subject_id')
    fig = plt.figure(figsize=(12, 8))
    ax_weight = fig.add_subplot(3, 1, 1)
    ax_water = fig.add_subplot(3, 1, 2, sharex=ax_weight)
    ax_weight_real = fig.add_subplot(3, 1, 3, sharex=ax_weight)
    for subject_id in mice_on_wr:
        water_restriction_log = lab.WaterRestriction().WaterRestrictionLog() & {'subject_id': subject_id}
        if len(water_restriction_log) == 0:
            continue
        baseline_weight = (lab.WaterRestriction() & {'subject_id': subject_id}).fetch1('wr_start_weight')
        dates = water_restriction_log.fetch('log_datetime')
        weights = np.asarray(water_restriction_log.fetch('weight'), float)
        weights_after = np.asarray(water_restriction_log.fetch('weight_after_watering'), float)
        water_consumed = weights_after - weights
        relative_weights = weights / float(baseline_weight) * 100
        linewidth = 2.5 if subject_id == highlight_subject_id else 1
        ax_weight.plot(dates, relative_weights, 'o-', label=subject_id, linewidth=linewidth)
        ax_water.plot(dates, water_consumed, 'o-', label=subject_id, linewidth=linewidth)
        ax_weight_real.plot(dates, weights, 'o-', label=subject_id, linewidth=linewidth)

    ax_weight.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_weight.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    for label in ax_weight_real.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    ax_weight.set_ylabel('Relative weight (%)')
    ax_water.set_ylabel('Water consumed (ml)')
    ax_weight_real.set_ylabel('Weight (g)')
    ax_weight_real.set_xlabel('date')
    ax_weight.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    fig.tight_layout()

    if fig_dir is not None:
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(os.path.join(fig_dir, 'water_restriction_overview.png'), dpi=150)
    return fig


def plot_trials_per_mouse(lab, experiment, subject_ids=None, fig_dir=None):
    """Bar chart of total trial count per subject that has run at least one trial.

    If subject_ids is given, only those subjects are shown (still restricted to ones
    with at least one trial); otherwise every subject with trials is shown.
    If fig_dir is given, saves to '{fig_dir}/trials_per_mouse.png'.
    Returns the figure.
    """
    mouse_ids = np.sort(lab.Subject.fetch('subject_id'))
    trial_nums = np.array([len(experiment.SessionTrial() & {'subject_id': m}) for m in mouse_ids])
    keep = trial_nums > 0
    if subject_ids:
        keep = keep & np.isin(mouse_ids, list(subject_ids))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(mouse_ids[keep], trial_nums[keep])
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    ax.set_xlabel('Mouse ID')
    ax.set_ylabel('Number of Trials')
    fig.tight_layout()

    if fig_dir is not None:
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(os.path.join(fig_dir, 'trials_per_mouse.png'), dpi=150)
    return fig


def _fetch_session_trend(experiment, subject_id):
    """Per-session (trial count, session length, hit rate) arrays for one subject, sorted by session."""
    available_sessions = np.sort((experiment.Session() & {'subject_id': subject_id}).fetch('session'))
    trial_nums, hit_rates, session_lengths = [], [], []
    for session in available_sessions:
        trials = (experiment.SessionTrial() & {'subject_id': subject_id, 'session': session}).fetch('trial')
        trial_nums.append(len(trials))
        hits = len(experiment.BehaviorTrial() & {'subject_id': subject_id, 'session': session, 'outcome': 'hit'})
        hit_rates.append(hits / len(trials) if len(trials) > 0 else np.nan)
        if len(trials) > 0:
            session_length = (experiment.SessionTrial() & {'subject_id': subject_id, 'session': session,
                                                             'trial': int(np.max(trials))}).fetch1('trial_end_time')
        else:
            session_length = np.nan
        session_lengths.append(session_length)
    return available_sessions, np.array(trial_nums), np.array(session_lengths, dtype=float), np.array(hit_rates, dtype=float)


def plot_subject_behavior_trend(experiment, subject_id, fig_dir=None):
    """Per-session trial count, session length, and hit rate for one subject, across all its sessions.

    If fig_dir is given, saves to '{fig_dir}/{subject_id}_behavior_trend.png'.
    Returns the figure.
    """
    available_sessions, trial_nums, session_lengths, hit_rates = _fetch_session_trend(experiment, subject_id)

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(available_sessions, trial_nums, 'o-')
    ax1.set_ylabel('number of trials')
    ax1.set_title(f'{subject_id} behavior overview')
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
    ax2.plot(available_sessions, session_lengths, 'o-')
    ax2.set_ylabel('session length (s)')
    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)
    ax3.plot(available_sessions, hit_rates, 'o-')
    ax3.set_ylabel('hit rate')
    ax3.set_xlabel('session')
    fig.tight_layout()

    if fig_dir is not None:
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(os.path.join(fig_dir, f'{subject_id}_behavior_trend.png'), dpi=150)
    return fig


def plot_subject_behavior_trends(experiment, subject_ids, fig_dir=None):
    """Per-session trial count, session length, and hit rate for multiple subjects, overlaid.

    Each subject gets its own color and a legend entry. If subject_ids is empty, returns
    an empty placeholder figure (nothing to fetch/plot).
    If fig_dir is given, saves to '{fig_dir}/subject_behavior_trends.png'.
    Returns the figure.
    """
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)

    if subject_ids:
        colors = plt.cm.tab10(np.arange(len(subject_ids)) % 10)
        for subject_id, color in zip(subject_ids, colors):
            sessions, trial_nums, session_lengths, hit_rates = _fetch_session_trend(experiment, subject_id)
            ax1.plot(sessions, trial_nums, 'o-', color=color, label=subject_id)
            ax2.plot(sessions, session_lengths, 'o-', color=color)
            ax3.plot(sessions, hit_rates, 'o-', color=color)
        ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'Select one or more mice to show session trends',
                  ha='center', va='center', transform=ax2.transAxes, color='gray')

    ax1.set_ylabel('number of trials')
    ax2.set_ylabel('session length (s)')
    ax3.set_ylabel('hit rate')
    ax3.set_xlabel('session')
    fig.tight_layout()

    if fig_dir is not None:
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(os.path.join(fig_dir, 'subject_behavior_trends.png'), dpi=150)
    return fig
