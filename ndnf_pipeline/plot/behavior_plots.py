import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.ndimage import uniform_filter1d, median_filter, gaussian_filter1d
from scipy.signal import savgol_filter

DEFAULT_UNIFORM_FORCE_RANGE = np.asarray([-1, 1, -1, 1]) * 10
ALL_EPOCHS = ('quiescence', 'response', 'reward')
FILTER_METHODS = ('none', 'boxcar', 'median', 'gaussian', 'savgol')


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


def _filter_trace(values, method, window=5, sigma=2.0, polyorder=3):
    """Smooth one trial's 1D trace with the given method ('none'/'boxcar'/'median'/'gaussian'/
    'savgol'); 'nearest'-padded at the edges so the output stays the same length as the input.

    boxcar/median/savgol take `window` (samples, forced odd where the underlying filter
    requires it); gaussian takes `sigma` (samples); savgol additionally takes `polyorder`
    (capped below `window` so it stays fittable, and below the trace length for short trials).
    """
    values = np.asarray(values, dtype=float)
    if not method or method == 'none' or values.size == 0:
        return values
    if method == 'boxcar':
        return uniform_filter1d(values, size=max(1, int(window)), mode='nearest')
    if method == 'median':
        size = max(1, int(window))
        if size % 2 == 0:
            size += 1
        return median_filter(values, size=size, mode='nearest')
    if method == 'gaussian':
        return gaussian_filter1d(values, sigma=max(1e-6, float(sigma)), mode='nearest')
    if method == 'savgol':
        size = max(3, int(window))
        if size % 2 == 0:
            size += 1
        if size > values.size:
            size = values.size if values.size % 2 == 1 else values.size - 1
        if size < 3:
            return values  # trial too short to fit a polynomial window meaningfully
        order = min(max(0, int(polyorder)), size - 1)
        return savgol_filter(values, window_length=size, polyorder=order, mode='nearest')
    raise ValueError(f"unknown filter method: {method!r}")


def _filter_traces(lr_traces, pa_traces, method, **filter_params):
    """Apply _filter_trace to every trial's lr/pa trace; a no-op when method is 'none'/None."""
    if not method or method == 'none':
        return lr_traces, pa_traces
    filtered_lr = [_filter_trace(t, method, **filter_params) for t in lr_traces]
    filtered_pa = [_filter_trace(t, method, **filter_params) for t in pa_traces]
    return filtered_lr, filtered_pa


def _estimate_sample_interval(times_list):
    """Median force-trace sample interval (seconds), pooling sample-to-sample diffs across
    every trial's force_trace_time array. None if there isn't enough data to tell (e.g. every
    trial has 0 or 1 samples)."""
    diffs = [np.diff(np.asarray(t, float)) for t in times_list if len(t) > 1]
    if not diffs:
        return None
    return float(np.median(np.concatenate(diffs)))


def ms_to_samples(duration_ms, sample_interval_s, minimum=1):
    """Convert a duration in milliseconds to an integer number of samples, given the sample
    interval in seconds. Falls back to `minimum` if the sample interval isn't known (None/<=0)."""
    if not sample_interval_s or sample_interval_s <= 0:
        return minimum
    return max(minimum, int(round(duration_ms / 1000.0 / sample_interval_s)))


def ms_to_samples_float(duration_ms, sample_interval_s, minimum=1e-6):
    """Like ms_to_samples, but keeps a fractional sample count (for e.g. Gaussian sigma, which
    doesn't need to be an integer)."""
    if not sample_interval_s or sample_interval_s <= 0:
        return minimum
    return max(minimum, duration_ms / 1000.0 / sample_interval_s)


def estimate_force_sample_interval(subject_id, session, block):
    """Median force-trace sample interval (seconds) for one block, estimated from a single
    trial's timestamps. Cheap enough to call live from a GUI (e.g. to preview how many samples
    a millisecond-based filter parameter works out to) without building a full figure. Returns
    None if the block has no force trace yet.
    """
    from ndnf_pipeline import experiment

    times = (experiment.TrialForceTrace() * experiment.BehaviorTrial()
             & {'subject_id': subject_id, 'session': session, 'block': block}).fetch('force_trace_time', limit=1)
    if len(times) == 0:
        return None
    t = np.asarray(times[0], float)
    return float(np.median(np.diff(t))) if t.size > 1 else None


def _force_histogram(lr_traces, pa_traces, histrange, bins=50):
    # log-density 2D histogram of force, with -inf (empty bins) clipped to the finite minimum
    try:
        lr_cat = np.concatenate(lr_traces)
        pa_cat = np.concatenate(pa_traces)
        valid = ~(np.isnan(lr_cat) | np.isnan(pa_cat))  # epoch-filtered traces carry NaN gaps
        forcehist, binx, biny = np.histogram2d(
            lr_cat[valid], pa_cat[valid],
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


def _fetch_quiescence_response(experiment, key):
    """Per-trial quiescence and response durations for a subject/session/block key.

    quiescence = trial start -> response period start (the 'go' event)
    response   = response period start -> threshold crossing
    Only trials with both events are included. Returns (trial_numbers, quiescence_values,
    response_values) as arrays, aligned by index.
    """
    rewarded_trial, time_to_reward = (experiment.TrialEvent() * experiment.BehaviorTrial() * experiment.Block()
                                       & {**key, 'trial_event_type': 'threshold crossing'}).fetch('trial', 'trial_event_time')
    go_trial, go_time = (experiment.TrialEvent() * experiment.BehaviorTrial() * experiment.Block()
                          & {**key, 'trial_event_type': 'go'}).fetch('trial', 'trial_event_time')
    go_time_by_trial = dict(zip(go_trial.tolist(), np.asarray(go_time, float)))
    trials, quiescence_value, response_value = [], [], []
    for trial, t2r in zip(rewarded_trial, np.asarray(time_to_reward, float)):
        g = go_time_by_trial.get(int(trial))
        if g is None:
            continue
        trials.append(trial)
        quiescence_value.append(g)
        response_value.append(t2r - g)
    return np.array(trials), np.array(quiescence_value), np.array(response_value)


def _event_time_by_trial(experiment, key, event_type):
    """{trial: trial_event_time} for every trial in a block that has this event type."""
    ev_trial, ev_time = (experiment.TrialEvent() * experiment.BehaviorTrial() * experiment.Block()
                          & {**key, 'trial_event_type': event_type}).fetch('trial', 'trial_event_time')
    return dict(zip(ev_trial.tolist(), np.asarray(ev_time, float)))


def _epoch_windows_from_events(go_by_trial, threshold_by_trial, reward_by_trial,
                                trial_nums, trial_starts, trial_ends):
    """Per-trial (quiescence, response, reward) time windows, relative to trial start.

    quiescence = trial start -> 'go' event; response = 'go' -> 'threshold crossing' (falling
    back to trial end if the trial has no threshold crossing, e.g. a miss); reward =
    'reward' event (falling back to 'threshold crossing') -> trial end. A trial missing the
    event(s) an epoch depends on simply has no entry for that epoch.

    Returns {trial: {epoch_name: (start, end)}}.
    """
    windows = {}
    for trial, start, end in zip(trial_nums, np.asarray(trial_starts, float), np.asarray(trial_ends, float)):
        trial = int(trial)
        duration = end - start
        window = {}
        go_t = go_by_trial.get(trial)
        threshold_t = threshold_by_trial.get(trial)
        reward_t = reward_by_trial.get(trial)
        if go_t is not None:
            window['quiescence'] = (0.0, go_t)
            window['response'] = (go_t, threshold_t if threshold_t is not None else duration)
        reward_start = reward_t if reward_t is not None else threshold_t
        if reward_start is not None:
            window['reward'] = (reward_start, duration)
        windows[trial] = window
    return windows


def _fetch_epoch_windows(experiment, key):
    """_epoch_windows_from_events, fetching its inputs itself for every trial in a block/session key."""
    go_by_trial = _event_time_by_trial(experiment, key, 'go')
    threshold_by_trial = _event_time_by_trial(experiment, key, 'threshold crossing')
    reward_by_trial = _event_time_by_trial(experiment, key, 'reward')
    trial_nums, trial_starts, trial_ends = (experiment.SessionTrial() * experiment.BehaviorTrial() * experiment.Block()
                                             & key).fetch('trial', 'trial_start_time', 'trial_end_time')
    return _epoch_windows_from_events(go_by_trial, threshold_by_trial, reward_by_trial,
                                       trial_nums, trial_starts, trial_ends)


def _mask_trace_by_epochs(times, values, window, epochs):
    """Blank out (set to NaN) samples of `values` that fall outside every requested epoch.

    `times`/`values` are one trial's samples (relative to trial start); `window` is that
    trial's dict from _fetch_epoch_windows (missing epochs contribute nothing, so a trial
    lacking e.g. a reward event is fully excluded when only 'reward' is requested).
    """
    times = np.asarray(times, float)
    values = np.array(values, dtype=float, copy=True)
    mask = np.zeros(times.shape, dtype=bool)
    for epoch in epochs:
        window_range = window.get(epoch)
        if window_range is None:
            continue
        lo, hi = window_range
        mask |= (times >= lo) & (times <= hi)
    values[~mask] = np.nan
    return values


def _filter_traces_by_epochs(trial_nums, times_list, lr_traces, pa_traces, windows, epochs):
    """Apply _mask_trace_by_epochs to every trial's lr/pa trace, unless epochs is None/empty
    or already covers every epoch (in which case nothing is filtered, for cheap backward
    compatibility with 'show everything')."""
    if not epochs or set(epochs) >= set(ALL_EPOCHS):
        return lr_traces, pa_traces
    filtered_lr, filtered_pa = [], []
    for trial, times, lr, pa in zip(trial_nums, times_list, lr_traces, pa_traces):
        window = windows.get(int(trial), {})
        filtered_lr.append(_mask_trace_by_epochs(times, lr, window, epochs))
        filtered_pa.append(_mask_trace_by_epochs(times, pa, window, epochs))
    return filtered_lr, filtered_pa


def _plot_quiescence_response(ax, trials, quiescence_value, response_value, color,
                               smoothing_window=10, log_yscale=False, label=None):
    """Plot quiescence ('^') and response ('.') durations per trial, plus rolling means, on ax."""
    ax.plot(trials, quiescence_value, '^', color=color, markersize=4, alpha=0.6, label=label)
    ax.plot(trials, response_value, '.', color=color, alpha=0.6)
    if len(trials) > smoothing_window >= 1:
        window = np.ones(smoothing_window) / smoothing_window
        smoothed_trials = np.convolve(trials, window, mode='valid')
        ax.plot(smoothed_trials, np.convolve(quiescence_value, window, mode='valid'), '-', color=color)
        ax.plot(smoothed_trials, np.convolve(response_value, window, mode='valid'), '--', color=color)
    ax.set_xlabel('trial#')
    ax.set_ylabel('time (s)')
    ax.set_yscale('log' if log_yscale else 'linear')


def _add_quiescence_response_marker_legend(ax, smoothing_window, bbox_to_anchor, loc='upper right', ncol=2):
    # marker and line handles are kept separate (rather than combined on one handle) since a
    # marker overlaid on a dashed line is hard to read at small font sizes
    handles = [
        Line2D([0], [0], marker='^', linestyle='None', color='black', label='quiescence (per trial)'),
        Line2D([0], [0], marker='None', linestyle='-', color='black', label=f'{smoothing_window}-trial rolling mean'),
        Line2D([0], [0], marker='.', linestyle='None', color='black', label='response (per trial)'),
        Line2D([0], [0], marker='None', linestyle='--', color='black', label=f'{smoothing_window}-trial rolling mean'),
    ]
    ax.legend(handles=handles, loc=loc, bbox_to_anchor=bbox_to_anchor, fontsize=7, ncol=ncol)


def plot_block_force_figure(subject_id, session, block,
                             force_uniform_range=True, uniform_force_range=None,
                             perf_log_yscale=False, perf_smoothing_window=10,
                             trials=None, epochs=None,
                             filter_method=None, filter_window_ms=50.0, filter_sigma_ms=20.0,
                             filter_polyorder=3, fig_dir=None):
    """6-panel figure for one block: target LUT, performance, force distribution, force
    trajectories (spatial), lickport position vs. time, and force vs. time.

    X is always Left-Right (L<0, R>0) and Y is always Posterior-Anterior (P<0, A>0),
    regardless of which raw loadcell axis/direction the rig recorded.

    The performance panel (quiescence/response duration per trial) always covers every
    trial in the block. If trials is given, the force distribution, spatial trajectory,
    lickport, and force-vs-time panels are restricted to just those trial numbers;
    otherwise they use every trial in the block.

    epochs restricts the force distribution, spatial trajectory, and force-vs-time panels
    to samples falling within the given trial epoch(s): any subset of ('quiescence',
    'response', 'reward'). None (or a set covering all three) means no restriction.

    filter_method smooths each trial's force trace (per-trial, before the epoch restriction
    above) with one of FILTER_METHODS: None/'none' (no filtering), 'boxcar' (moving average,
    `filter_window_ms`), 'median' (`filter_window_ms`), 'gaussian' (`filter_sigma_ms`), or
    'savgol' (Savitzky-Golay polynomial fit, `filter_window_ms` and `filter_polyorder`).
    `filter_window_ms`/`filter_sigma_ms` are converted to samples using this block's own force
    trace sample interval (so they mean the same thing regardless of sampling rate). Filtering
    feeds every downstream panel that uses the force traces (distribution, trajectory,
    force-vs-time).

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

    # period boundaries for every trial in the block (not just the plotted trials), used both to
    # shade the lickport/force-vs-time panels below and (via _epoch_windows_from_events) to
    # restrict the histogram/trajectory/force-vs-time panels to the requested epoch(s); 'go' marks
    # the end of quiescence/start of response, 'reward' (falling back to 'threshold crossing' if
    # this trial has no separate reward event) marks reward delivery
    go_by_trial = _event_time_by_trial(experiment, key, 'go')
    threshold_by_trial = _event_time_by_trial(experiment, key, 'threshold crossing')
    reward_by_trial = _event_time_by_trial(experiment, key, 'reward')
    all_trial_nums, all_trial_starts, all_trial_ends = (
        experiment.SessionTrial() * experiment.BehaviorTrial() * experiment.Block() & key
    ).fetch('trial', 'trial_start_time', 'trial_end_time')
    epoch_windows = _epoch_windows_from_events(go_by_trial, threshold_by_trial, reward_by_trial,
                                                all_trial_nums, all_trial_starts, all_trial_ends)

    fig = plt.figure(figsize=(12, 19))
    gs = plt.GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 0.6, 0.8], hspace=0.45, wspace=0.3)
    ax_target_LUT = fig.add_subplot(gs[0, 0])

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

    ax_performance = fig.add_subplot(gs[1, 0])
    perf_trials, quiescence_value, response_value = _fetch_quiescence_response(experiment, key)
    _plot_quiescence_response(ax_performance, perf_trials, quiescence_value, response_value,
                               color='tab:blue', smoothing_window=perf_smoothing_window,
                               log_yscale=perf_log_yscale)
    _add_quiescence_response_marker_legend(ax_performance, perf_smoothing_window, bbox_to_anchor=(1.0, -0.2))

    ax_force_hist = fig.add_subplot(gs[0, 1])
    force_traces_0_query = (experiment.TrialForceTrace() * experiment.TrialForceTrace.TrialForceAxis()
                             * experiment.BehaviorTrial() * experiment.Block() & {**key, 'force_axis_idx': 0})
    force_traces_1_query = (experiment.TrialForceTrace.TrialForceAxis() * experiment.BehaviorTrial() * experiment.Block()
                             & {**key, 'force_axis_idx': 1})
    if trials:
        trial_restriction = [{'trial': int(t)} for t in trials]
        force_traces_0_query = force_traces_0_query & trial_restriction
        force_traces_1_query = force_traces_1_query & trial_restriction
    force_trials, force_trace_times, force_traces_0_ = force_traces_0_query.fetch(
        'trial', 'force_trace_time', 'force_trace_value', order_by='trial')
    _, force_traces_1_ = force_traces_1_query.fetch('trial', 'force_trace_value', order_by='trial')
    f0_baseline = np.median(np.concatenate(force_traces_0_))
    f1_baseline = np.median(np.concatenate(force_traces_1_))
    force_traces_0 = [f - f0_baseline for f in force_traces_0_]
    force_traces_1 = [f - f1_baseline for f in force_traces_1_]

    lr_traces, pa_traces = _lr_pa_traces(force_traces_0, force_traces_1, lr_idx, pa_idx, lr_sign, pa_sign)
    sample_interval_s = _estimate_sample_interval(force_trace_times)
    filter_window = ms_to_samples(filter_window_ms, sample_interval_s, minimum=1)
    filter_sigma = ms_to_samples_float(filter_sigma_ms, sample_interval_s, minimum=1e-6)
    lr_traces, pa_traces = _filter_traces(lr_traces, pa_traces, filter_method, window=filter_window,
                                           sigma=filter_sigma, polyorder=filter_polyorder)
    lr_traces, pa_traces = _filter_traces_by_epochs(force_trials, force_trace_times, lr_traces, pa_traces,
                                                     epoch_windows, epochs)

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

    ax_traj = fig.add_subplot(gs[1, 1])
    _plot_force_trajectories(ax_traj, lr_traces, pa_traces, force_uniform_range, uniform_force_range)

    # --- shared per-trial timing, used by both the lickport and force-vs-time panels below ---
    time_query = (experiment.TrialForceTrace() * experiment.BehaviorTrial() * experiment.Block()
                  * experiment.SessionTrial() & key)
    if trials:
        time_query = time_query & trial_restriction
    time_trials, trial_starts, trial_ends, force_times = time_query.fetch(
        'trial', 'trial_start_time', 'trial_end_time', 'force_trace_time', order_by='trial')
    if not np.array_equal(time_trials, force_trials):
        raise RuntimeError('trial mismatch between TrialForceTrace and TrialForceTrace.TrialForceAxis fetches')
    trial_start_by_num = dict(zip(time_trials.tolist(), np.asarray(trial_starts, float)))
    trial_end_by_num = dict(zip(time_trials.tolist(), np.asarray(trial_ends, float)))

    # go_by_trial/threshold_by_trial/reward_by_trial were already fetched above (for epoch_windows)

    def _shade_trial_periods(ax, trial, trial_start, trial_end):
        go_t = go_by_trial.get(trial)
        threshold_t = threshold_by_trial.get(trial)
        reward_t = reward_by_trial.get(trial)
        if go_t is not None:
            quiescence_end = trial_start + go_t
            response_end = trial_start + threshold_t if threshold_t is not None else trial_end
            ax.axvspan(trial_start, quiescence_end, color='gray', alpha=0.15, linewidth=0)
            ax.axvspan(quiescence_end, response_end, color='gold', alpha=0.15, linewidth=0)
        reward_time_abs = trial_start + reward_t if reward_t is not None else (
            trial_start + threshold_t if threshold_t is not None else None)
        if reward_time_abs is not None:
            ax.axvline(reward_time_abs, color='green', linewidth=1.2, alpha=0.8)

    # --- lickport position vs. time, shares its x axis with the force-vs-time panel below ---
    ax_force_time = fig.add_subplot(gs[3, :])
    ax_lickport = fig.add_subplot(gs[2, :], sharex=ax_force_time)
    lickport_query = (experiment.TrialRewardPortPosition() * experiment.BehaviorTrial() * experiment.Block()
                       * experiment.SessionTrial() & key)
    if trials:
        lickport_query = lickport_query & trial_restriction
    lp_trials, lp_trial_starts, lp_times, lp_values = lickport_query.fetch(
        'trial', 'trial_start_time', 'reward_port_position_time', 'reward_port_position_values', order_by='trial')
    lickport_carry = None  # last known lickport position, carried across the trial-to-trial gap
    for trial, trial_start, lp_time, lp_val in zip(lp_trials, np.asarray(lp_trial_starts, float), lp_times, lp_values):
        trial = int(trial)
        trial_end = trial_end_by_num.get(trial, trial_start)
        _shade_trial_periods(ax_lickport, trial, trial_start, trial_end)
        lp_time_arr = np.asarray(lp_time, float)
        lp_val_arr = np.asarray(lp_val, float)
        # the port stays closed (wherever the last reward left it) through the whole quiescence
        # period -- nothing touches it, and it isn't logged again until the 'go' event opens it
        # for the response period -- so hold the last known value (carried over from the previous
        # trial if this trial hasn't logged anything yet) until that first real sample, rather than
        # letting a plain connecting line ramp gradually across the silent gap
        if len(lp_time_arr) == 0:
            if lickport_carry is not None:
                ax_lickport.step([trial_start, trial_end], [lickport_carry, lickport_carry], where='post',
                                  color='tab:purple', linewidth=0.8, alpha=0.8)
            continue
        lead_val = lickport_carry if lickport_carry is not None else lp_val_arr[0]
        ext_time_abs = np.concatenate([[trial_start], trial_start + lp_time_arr, [trial_end]])
        ext_val = np.concatenate([[lead_val], lp_val_arr, [lp_val_arr[-1]]])
        ax_lickport.step(ext_time_abs, ext_val, where='post', color='tab:purple', linewidth=0.8, alpha=0.8)
        lickport_carry = lp_val_arr[-1]

    lick_query = (experiment.TrialEvent() * experiment.BehaviorTrial() * experiment.Block()
                  & {**key, 'trial_event_type': 'lick'})
    if trials:
        lick_query = lick_query & trial_restriction
    lick_trials_raw, lick_times_raw = lick_query.fetch('trial', 'trial_event_time')
    lick_abs_times = [trial_start_by_num[int(t)] + float(lt) for t, lt in zip(lick_trials_raw, lick_times_raw)
                       if int(t) in trial_start_by_num]
    if lick_abs_times:
        y0, y1 = ax_lickport.get_ylim()
        tick_height = 0.08 * (y1 - y0)
        ax_lickport.eventplot(lick_abs_times, orientation='horizontal', lineoffsets=y1 - tick_height / 2,
                               linelengths=tick_height, colors='black')
    ax_lickport.set_ylabel('lickport pos (mm)')
    plt.setp(ax_lickport.get_xticklabels(), visible=False)  # x axis is shared with the panel below

    # --- the same (baseline-corrected, axis-remapped) LR/PA traces used above, plotted against
    # absolute session time instead of space, one line per trial so gaps between non-adjacent/
    # unselected trials aren't bridged by a spurious connecting line ---
    for trial, trial_start, trial_end, f_time, lr, pa in zip(
            time_trials, np.asarray(trial_starts, float), np.asarray(trial_ends, float), force_times, lr_traces, pa_traces):
        trial = int(trial)
        _shade_trial_periods(ax_force_time, trial, trial_start, trial_end)
        abs_time = trial_start + np.asarray(f_time, float)
        ax_force_time.plot(abs_time, lr, '-', color='tab:blue', linewidth=0.7, alpha=0.7)
        ax_force_time.plot(abs_time, pa, '-', color='tab:orange', linewidth=0.7, alpha=0.7)
    if force_uniform_range:
        ax_force_time.set_ylim(min(uniform_force_range[:2].min(), uniform_force_range[2:].min()),
                                max(uniform_force_range[:2].max(), uniform_force_range[2:].max()))
    ax_force_time.set_xlabel('session time (s)')
    ax_force_time.set_ylabel('force (g)')
    fig.subplots_adjust(bottom=0.08)
    ax_force_time.legend(handles=[
        Line2D([0], [0], color='tab:blue', label='Left - Right'),
        Line2D([0], [0], color='tab:orange', label='Posterior - Anterior'),
        Line2D([0], [0], color='tab:purple', label='lickport pos'),
        Line2D([0], [0], color='black', marker='|', linestyle='None', markersize=8, label='licks'),
        Patch(color='gray', alpha=0.3, label='quiescence'),
        Patch(color='gold', alpha=0.3, label='response'),
        Line2D([0], [0], color='green', label='reward'),
    ], loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8, ncol=7)

    if fig_dir is not None:
        os.makedirs(fig_dir, exist_ok=True)
        fig.savefig(os.path.join(fig_dir, f'{subject_id}_s{session}_b{block}.png'), dpi=150)

    return fig


def plot_session_blocks_overview(subject_id, session,
                                  force_uniform_range=True, uniform_force_range=None,
                                  perf_log_yscale=False, perf_smoothing_window=10,
                                  epochs=None,
                                  filter_method=None, filter_window_ms=50.0, filter_sigma_ms=20.0,
                                  filter_polyorder=3, fig_dir=None):
    """Session-level overview across all of a session's blocks.

    Row 1: for every rewarded trial in the session (one panel), the quiescence duration
           (trial start -> response period start, '^') and response duration (response
           period start -> threshold crossing, '.'), with each block's trial range
           highlighted and color-coded. Y axis is log-scaled if perf_log_yscale is True.
    Row 2: each block's target LUT, side by side, sharing the same axes and colormap.
    Row 3: each block's force distribution histogram, side by side, sharing the
           same axes and colormap.
    Row 4: each block's force trajectories (early=blue, late=red), side by side.

    X is always Left-Right (L<0, R>0) and Y is always Posterior-Anterior (P<0, A>0).

    epochs restricts rows 3 and 4 (histogram, trajectories) to samples falling within the
    given trial epoch(s): any subset of ('quiescence', 'response', 'reward'). None (or a
    set covering all three) means no restriction.

    filter_method smooths each trial's force trace (per-trial, per-block, before the epoch
    restriction above) with one of FILTER_METHODS: None/'none' (no filtering), 'boxcar'
    (moving average, `filter_window_ms`), 'median' (`filter_window_ms`), 'gaussian'
    (`filter_sigma_ms`), or 'savgol' (Savitzky-Golay polynomial fit, `filter_window_ms` and
    `filter_polyorder`). `filter_window_ms`/`filter_sigma_ms` are converted to samples using
    each block's own force trace sample interval.

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

    # --- gather everything needed per block up front, so rows 2 and 3 can share vmin/vmax ---
    block_data = []
    for block in available_blocks:
        key = {'subject_id': subject_id, 'session': session, 'block': block}
        task_setting_id, target_force_lut = (experiment.TaskSettings() * experiment.Block() & key).fetch1('task_setting_id', 'target_force_lut')
        force_axes_dict, lr_idx, pa_idx, lr_sign, pa_sign = _get_block_force_axes(experiment, subject_id, session, task_setting_id)
        lut, lut_extent, lr_extent, pa_extent = _normalize_lut(target_force_lut, force_axes_dict, lr_idx, pa_idx, lr_sign, pa_sign)

        force_traces_0_query = (experiment.TrialForceTrace() * experiment.TrialForceTrace.TrialForceAxis()
                                 * experiment.BehaviorTrial() * experiment.Block() & {**key, 'force_axis_idx': 0})
        force_trials, force_trace_times, force_traces_0_ = force_traces_0_query.fetch(
            'trial', 'force_trace_time', 'force_trace_value', order_by='trial')
        force_traces_1_ = (experiment.TrialForceTrace.TrialForceAxis() * experiment.BehaviorTrial() * experiment.Block()
                            & {**key, 'force_axis_idx': 1}).fetch('force_trace_value', order_by='trial')
        f0_baseline = np.median(np.concatenate(force_traces_0_))
        f1_baseline = np.median(np.concatenate(force_traces_1_))
        force_traces_0 = [f - f0_baseline for f in force_traces_0_]
        force_traces_1 = [f - f1_baseline for f in force_traces_1_]
        lr_traces, pa_traces = _lr_pa_traces(force_traces_0, force_traces_1, lr_idx, pa_idx, lr_sign, pa_sign)
        sample_interval_s = _estimate_sample_interval(force_trace_times)
        filter_window = ms_to_samples(filter_window_ms, sample_interval_s, minimum=1)
        filter_sigma = ms_to_samples_float(filter_sigma_ms, sample_interval_s, minimum=1e-6)
        lr_traces, pa_traces = _filter_traces(lr_traces, pa_traces, filter_method, window=filter_window,
                                               sigma=filter_sigma, polyorder=filter_polyorder)
        epoch_windows = _fetch_epoch_windows(experiment, key)
        lr_traces, pa_traces = _filter_traces_by_epochs(force_trials, force_trace_times, lr_traces, pa_traces,
                                                         epoch_windows, epochs)

        histrange = [uniform_force_range[:2], uniform_force_range[2:]] if force_uniform_range else [lr_extent, pa_extent]
        forcehist, binx, biny = _force_histogram(lr_traces, pa_traces, histrange)

        block_trials = (experiment.BehaviorTrial() & key).fetch('trial')
        quiescence_trial, quiescence_value, response_value = _fetch_quiescence_response(experiment, key)

        block_data.append(dict(block=block, lut=lut, lut_extent=lut_extent,
                                forcehist=forcehist, binx=binx, biny=biny,
                                lr_traces=lr_traces, pa_traces=pa_traces,
                                block_trials=block_trials,
                                quiescence_trial=quiescence_trial,
                                quiescence_value=quiescence_value,
                                response_value=response_value))

    lut_vmin = min(np.min(bd['lut']) for bd in block_data)
    lut_vmax = max(np.max(bd['lut']) for bd in block_data)
    hist_vals = [bd['forcehist'] for bd in block_data if bd['forcehist'] is not None]
    hist_vmin = min(np.min(h) for h in hist_vals) if hist_vals else None
    hist_vmax = max(np.max(h) for h in hist_vals) if hist_vals else None

    fig = plt.figure(figsize=(4 * max(n_blocks, 1), 13))
    gs = plt.GridSpec(4, max(n_blocks, 1), figure=fig, height_ratios=[1, 1.1, 1.1, 1.1], hspace=0.5, wspace=0.3)

    # --- row 1: quiescence & response durations across the whole session, blocks highlighted ---
    ax_perf = fig.add_subplot(gs[0, :])
    for bd, color in zip(block_data, block_colors):
        if len(bd['block_trials']):
            ax_perf.axvspan(bd['block_trials'].min() - 0.5, bd['block_trials'].max() + 0.5, color=color, alpha=0.15)
        _plot_quiescence_response(ax_perf, bd['quiescence_trial'], bd['quiescence_value'], bd['response_value'],
                                   color=color, smoothing_window=perf_smoothing_window,
                                   log_yscale=perf_log_yscale)
    ax_perf.set_title(f'{subject_id} s{session}  {session_date} {session_time}')
    # shrink row 1's own axes and put the marker legend in the freed-up strip to its right,
    # so it never sits on top of the data (or of row 2 below it)
    pos = ax_perf.get_position()
    legend_width = 0.15
    ax_perf.set_position([pos.x0, pos.y0, pos.width - legend_width, pos.height])
    _add_quiescence_response_marker_legend(ax_perf, perf_smoothing_window, bbox_to_anchor=(1.02, 1.0),
                                            loc='upper left', ncol=1)

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
        if bi == 0:
            ax_hist.set_ylabel('Posterior - Anterior (g)')
        axes_hist.append(ax_hist)
    if im_hist is not None:
        fig.colorbar(im_hist, ax=axes_hist, label='fraction of time spent (log)')

    # --- row 4: force trajectories side by side ---
    # one shared colorbar for the whole row: each block's trajectories are already
    # colored by *fractional* trial position (0=first, 1=last), so a single colorbar
    # labeled first...last applies across blocks even though they have different trial counts
    axes_traj = []
    mid_col = max(n_blocks, 1) // 2
    for bi, (bd, color) in enumerate(zip(block_data, block_colors)):
        ax_traj = fig.add_subplot(gs[3, bi])
        _plot_force_trajectories(ax_traj, bd['lr_traces'], bd['pa_traces'], force_uniform_range, uniform_force_range,
                                  title=None, ylabel=False, add_colorbar=False)
        if bi != mid_col:
            ax_traj.set_xlabel('')
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
