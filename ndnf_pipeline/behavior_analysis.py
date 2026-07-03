import datajoint as dj
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from ndnf_pipeline.utils.pipeline_tools import get_schema_name
from ndnf_pipeline import experiment

schema = dj.schema(get_schema_name('behavior_analysis'), locals())



@schema
class BlockStatistics(dj.Computed):
    definition = """
    -> experiment.Block
    ---
    time_to_first_reward: decimal(9, 5)  # (s) threshold crossing time in the first rewarded trial
    time_to_last_rewards = null: decimal(9, 5)  # (s) median threshold crossing time over last 5 rewarded trials
    n_rewarded_trials: smallint          # number of rewarded trials in the block
    """

    class ExpFit(dj.Part):
        definition = """
        -> master
        ---
        first_trial_value: float  # (s) fitted time-to-reward at trial 0
        steady_state: float       # (s) asymptotic time-to-reward
        time_constant: float      # (trials) exponential time constant tau
        r_squared: float          # goodness of fit
        """

    class InvFit(dj.Part):
        definition = """
        -> master
        ---
        first_trial_value: float  # (s) fitted time-to-reward at trial 0
        steady_state: float       # (s) asymptotic time-to-reward
        time_constant: float      # (trials) 1/x time constant
        r_squared: float          # goodness of fit
        """

    def make(self, key):
        rewarded_trials, times_to_reward = (
            experiment.TrialEvent() * experiment.BehaviorTrial() * experiment.Block()
            & key & {'trial_event_type': 'threshold crossing'}
        ).fetch('trial', 'trial_event_time', order_by='trial')

        if len(rewarded_trials) == 0:
            return

        times_to_reward = times_to_reward.astype(float)

        # re-index trials from 0 relative to block start
        first_trial = int((experiment.BehaviorTrial() & key).fetch('trial').min())
        trial_in_block = (rewarded_trials - first_trial).astype(float)

        if len(rewarded_trials)<6:
            time_to_last_rewards = np.nan
        else:
            time_to_last_rewards = float(np.median(times_to_reward[-6:-1]))

        self.insert1({**key,
                      'time_to_first_reward': float(times_to_reward[0]),
                      'time_to_last_rewards': time_to_last_rewards,
                      'n_rewarded_trials': len(rewarded_trials)})

        if len(rewarded_trials) < 4:
            return

        t = trial_in_block
        y = times_to_reward
        p0 = [y[0], y[-1], max(len(t) / 3, 1.0)]
        bounds = ([0, 0, 0.1], [np.inf, np.inf, np.inf])

        def exp_decay(t, first_val, steady_state, tau):
            return steady_state + (first_val - steady_state) * np.exp(-t / tau)

        def inv_decay(t, first_val, steady_state, tau):
            return steady_state + (first_val - steady_state) * tau / (t + tau)

        for fit_fn, part_table in [(exp_decay, self.ExpFit()), (inv_decay, self.InvFit())]:
            try:
                popt, _ = curve_fit(fit_fn, t, y, p0=p0, bounds=bounds, maxfev=5000)
                y_pred = fit_fn(t, *popt)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
                part_table.insert1({**key,
                                    'first_trial_value': float(popt[0]),
                                    'steady_state': float(popt[1]),
                                    'time_constant': float(popt[2]),
                                    'r_squared': r_squared})
            except RuntimeError:
                pass




@schema
class TouchDetectionParams(dj.Lookup):
    definition = """
    touch_param_id: smallint
    ---
    smooth_window_time: float  # (s) boxcar smoothing window applied to each force axis
    sd_window_time: float      # (s) rolling SD window applied to force magnitude
    sd_threshold: float        # (g) SD threshold above which contact is detected
    force_threshold: float     # (g) smoothed force magnitude above which contact is always detected
    """
    contents = [
        {'touch_param_id': 0, 'smooth_window_time': 0.3, 'sd_window_time': 0.05,
         'sd_threshold': 0.05, 'force_threshold': 10.0}
    ]


@schema
class SaturationDetectionParams(dj.Lookup):
    definition = """
    sat_param_id: smallint
    ---
    smooth_window_time: float  # (s) boxcar smoothing window
    sd_window_time: float      # (s) rolling SD window
    force_threshold: float     # (g) |force| on a single axis must exceed this
    sd_threshold: float        # (g) rolling SD must be below this (flatline = saturated)
    """
    contents = [
        {'sat_param_id': 0, 'smooth_window_time': 0.01, 'sd_window_time': 0.01,
         'force_threshold': 120.0, 'sd_threshold': 0.05}
    ]


@schema
class TrialSaturationTimes(dj.Computed):
    definition = """
    -> experiment.TrialForceTrace
    -> SaturationDetectionParams
    ---
    n_saturation_epochs: smallint  # total across all axes
    """

    class SaturationEpoch(dj.Part):
        definition = """
        -> master
        -> experiment.TrialForceTrace.TrialForceAxis
        epoch_idx: smallint
        ---
        start_idx: int
        end_idx: int
        start_time: float    # (s) from trial start
        end_time: float      # (s) from trial start
        duration: float      # (s)
        """

    def make(self, key):
        params = (SaturationDetectionParams & key).fetch1()

        force_trace_time = (experiment.TrialForceTrace & key).fetch1('force_trace_time')
        sample_interval  = float(np.median(np.diff(force_trace_time)))

        smooth_window = max(1, int(params['smooth_window_time'] / sample_interval))
        sd_window     = max(2, int(params['sd_window_time'] / sample_interval))
        half_w        = smooth_window // 2

        axis_indices, axis_values = (experiment.TrialForceTrace.TrialForceAxis & key
                                      & 'force_axis_idx < 2').fetch(
            'force_axis_idx', 'force_trace_value', order_by='force_axis_idx'
        )

        all_epochs = []

        for force_axis_idx, raw in zip(axis_indices, axis_values):
            smoothed = np.convolve(raw, np.ones(smooth_window) / smooth_window, mode='valid')
            sd       = pd.Series(smoothed).rolling(window=sd_window).std()

            saturated = (
                (np.abs(smoothed) > params['force_threshold']) &
                (sd < params['sd_threshold']).fillna(False).values
            )

            smooth_time = force_trace_time[half_w: half_w + len(smoothed)]
            if len(smooth_time) < len(smoothed):
                smooth_time = force_trace_time[0] + np.arange(len(smoothed)) * sample_interval

            if not saturated.any():
                continue

            padded  = np.concatenate([[False], saturated, [False]]).astype(int)
            changes = np.diff(padded)
            starts  = np.where(changes ==  1)[0]
            ends    = np.where(changes == -1)[0] - 1

            for epoch_idx, (start, end) in enumerate(zip(starts, ends)):
                all_epochs.append({
                    **key,
                    'force_axis_idx': int(force_axis_idx),
                    'epoch_idx':      epoch_idx,
                    'start_idx':      int(start),
                    'end_idx':        int(end),
                    'start_time':     float(smooth_time[start]),
                    'end_time':       float(smooth_time[end]),
                    'duration':       float((end - start + 1) * sample_interval),
                })

        self.insert1({**key, 'n_saturation_epochs': len(all_epochs)})
        self.SaturationEpoch.insert(all_epochs)


@schema
class TrialTouchTimes(dj.Computed):
    definition = """
    -> TrialSaturationTimes
    -> TouchDetectionParams
    ---
    n_touch_epochs: smallint  # number of contiguous touch epochs detected
    """

    class TouchEpoch(dj.Part):
        definition = """
        -> master
        is_touch: bool       # True = touch, False = no touch
        epoch_idx: smallint  # index within touch or no-touch epochs separately
        ---
        start_idx: int       # start sample index into the smoothed trace (inclusive)
        end_idx: int         # end sample index into the smoothed trace (inclusive)
        start_time: float    # (s) from trial start
        end_time: float      # (s) from trial start
        duration: float      # (s)
        """

    def make(self, key):
        params = (TouchDetectionParams & key).fetch1()

        force_trace_time = (experiment.TrialForceTrace & key).fetch1('force_trace_time')
        if len(force_trace_time) == 0:
            print('empty trial')
            return
        sample_interval = float(np.median(np.diff(force_trace_time)))

        smooth_window = max(1, int(params['smooth_window_time'] / sample_interval))
        sd_window     = max(2, int(params['sd_window_time'] / sample_interval))

        axis_indices, axis_values = (experiment.TrialForceTrace.TrialForceAxis & key
                                      & 'force_axis_idx < 2').fetch(
            'force_axis_idx', 'force_trace_value', order_by='force_axis_idx'
        )
        force_axes = {int(idx): val for idx, val in zip(axis_indices, axis_values)}

        smoothed = {
            ax: np.convolve(force_axes[ax], np.ones(smooth_window) / smooth_window, mode='valid')
            for ax in force_axes
        }
        n_smooth = min(len(v) for v in smoothed.values())
        force_magnitude_smooth = np.sqrt(sum(smoothed[ax][:n_smooth] ** 2 for ax in smoothed))

        force_magnitude_smooth_sd = pd.Series(force_magnitude_smooth).rolling(window=sd_window).std()
        contact_times = (force_magnitude_smooth_sd > params['sd_threshold']).fillna(False).values

        # High force magnitude is always touch, regardless of SD
        contact_times |= (force_magnitude_smooth > params['force_threshold'])

        half_w = smooth_window // 2
        smooth_time = force_trace_time[half_w: half_w + n_smooth]
        if len(smooth_time) < n_smooth:
            smooth_time = force_trace_time[0] + np.arange(n_smooth) * sample_interval

        # Saturated windows on any channel are always touch
        sat_starts, sat_ends = (TrialSaturationTimes.SaturationEpoch & key).fetch(
            'start_time', 'end_time'
        )
        for sat_start, sat_end in zip(sat_starts, sat_ends):
            contact_times[
                (smooth_time >= float(sat_start)) & (smooth_time <= float(sat_end))
            ] = True

        if len(contact_times) == 0:
            self.insert1({**key, 'n_touch_epochs': 0})
            return

        changes = np.diff(contact_times.astype(int))
        block_starts = np.concatenate([[0], np.where(changes != 0)[0] + 1])
        block_ends = np.concatenate([np.where(changes != 0)[0], [len(contact_times) - 1]])

        epochs = []
        epoch_counters = {False: 0, True: 0}

        for start, end in zip(block_starts, block_ends):
            is_touch = bool(contact_times[start])
            duration = float((end - start + 1) * sample_interval)
            epochs.append({
                **key,
                'is_touch': is_touch,
                'epoch_idx': epoch_counters[is_touch],
                'start_idx': int(start),
                'end_idx': int(end),
                'start_time': float(smooth_time[start]),
                'end_time': float(smooth_time[end]),
                'duration': duration,
            })
            epoch_counters[is_touch] += 1

        self.insert1({**key, 'n_touch_epochs': epoch_counters[True]})
        self.TouchEpoch.insert(epochs)


@schema
class TrialTiming(dj.Computed):
    definition = """
    -> TrialTouchTimes
    ---
    first_touch_time: float          # (s) from session start, onset of first touch epoch
    trial_length: float              # (s) from first touch to trial end
    time_to_reward = null: float     # (s) from first touch to threshold crossing, null if unrewarded
    """

    @property
    def key_source(self):
        return TrialTouchTimes & 'n_touch_epochs > 0'

    def make(self, key):
        first_touch_in_trial = float(
            (TrialTouchTimes.TouchEpoch & key & {'is_touch': True}).fetch(
                'start_time', order_by='epoch_idx', limit=1
            )[0]
        )

        trial_start, trial_end = (experiment.SessionTrial & key).fetch1(
            'trial_start_time', 'trial_end_time'
        )
        first_touch_time = float(trial_start) + first_touch_in_trial
        trial_length = float(trial_end) - first_touch_time

        events = (experiment.TrialEvent & key & {'trial_event_type': 'threshold crossing'}).fetch(
            'trial_event_time', order_by='trial_event_id'
        )
        time_to_reward = float(events[0]) - first_touch_in_trial if len(events) > 0 else None

        self.insert1({
            **key,
            'first_touch_time': first_touch_time,
            'trial_length': trial_length,
            'time_to_reward': time_to_reward,
        })


@schema
class BlockTiming(dj.Computed):
    definition = """
    -> experiment.Block
    -> TouchDetectionParams
    -> SaturationDetectionParams
    ---
    block_start_time: float  # (s) from session start, first touch of first trial in block
    """

    @property
    def key_source(self):
        return (experiment.Block * TouchDetectionParams * SaturationDetectionParams
                & (TrialTiming * experiment.BehaviorTrial))

    def make(self, key):
        first_trial = int((experiment.BehaviorTrial & key).fetch('trial').min())
        block_start_time = float(
            (TrialTiming & {**key, 'trial': first_trial}).fetch1('first_touch_time')
        )
        self.insert1({**key, 'block_start_time': block_start_time})


# ---------------------------------------------------------------------------
# Submovement detection helpers
# ---------------------------------------------------------------------------

def _menger_curvature_radius(x, y, half_window=5):
    """Radius of curvature at each sample using the three-point Menger formula."""
    n = len(x)
    radius = np.full(n, np.inf)
    for i in range(half_window, n - half_window):
        x1, y1 = x[i - half_window], y[i - half_window]
        x2, y2 = x[i],               y[i]
        x3, y3 = x[i + half_window], y[i + half_window]
        a = np.hypot(x2 - x1, y2 - y1)
        b = np.hypot(x3 - x2, y3 - y2)
        c = np.hypot(x3 - x1, y3 - y1)
        area = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2
        if area > 1e-12:
            radius[i] = (a * b * c) / (4 * area)
    return radius


@schema
class SubmovementDetectionParams(dj.Lookup):
    definition = """
    subm_param_id: smallint
    ---
    smooth_window_time: float        # (s) boxcar smoothing for force traces before derivative
    curvature_half_window: int       # samples, half-window for Menger radius (each side)
    min_submovement_duration: float  # (s) minimum distance between segment boundaries
    speed_prominence: float          # minimum prominence of a speed local minimum (g/sample)
    cooccurrence_window_time: float  # (s) max lag between matched speed and curvature minima
    """
    contents = [
        {'subm_param_id': 0,
         'smooth_window_time': 0.03,
         'curvature_half_window': 5,
         'min_submovement_duration': 0.05,
         'speed_prominence': 0.05,
         'cooccurrence_window_time': 0.05}
    ]


@schema
class TrialSubmovements(dj.Computed):
    definition = """
    -> TrialTouchTimes
    -> SubmovementDetectionParams
    ---
    n_submovements: smallint
    """

    class Submovement(dj.Part):
        definition = """
        -> master
        submovement_idx: smallint
        ---
        start_time: float   # (s) from trial start
        end_time: float     # (s) from trial start
        duration: float     # (s)
        peak_speed: float   # (g/sample) max force-change speed during submovement
        """

    def make(self, key):
        from scipy.signal import find_peaks

        params = (SubmovementDetectionParams & key).fetch1()

        force_trace_time = (experiment.TrialForceTrace & key).fetch1('force_trace_time')
        if len(force_trace_time) < 3:
            self.insert1({**key, 'n_submovements': 0})
            return

        sample_interval = float(np.median(np.diff(force_trace_time)))
        smooth_window = max(1, int(params['smooth_window_time'] / sample_interval))
        min_dist      = max(2, int(params['min_submovement_duration'] / sample_interval))
        cooc_win      = max(1, int(params['cooccurrence_window_time'] / sample_interval))
        half_w        = smooth_window // 2
        curv_hw       = int(params['curvature_half_window'])

        axis_indices, axis_values = (
            experiment.TrialForceTrace.TrialForceAxis & key & 'force_axis_idx < 2'
        ).fetch('force_axis_idx', 'force_trace_value', order_by='force_axis_idx')

        if len(axis_values) < 2:
            self.insert1({**key, 'n_submovements': 0})
            return

        f_raw = {int(idx): val for idx, val in zip(axis_indices, axis_values)}
        f_smooth = {
            ax: np.convolve(f_raw[ax], np.ones(smooth_window) / smooth_window, mode='valid')
            for ax in f_raw
        }
        n = min(len(v) for v in f_smooth.values())
        f0, f1 = f_smooth[0][:n], f_smooth[1][:n]

        t = force_trace_time[half_w: half_w + n]
        if len(t) < n:
            t = force_trace_time[0] + np.arange(n) * sample_interval

        # Speed in force space (one sample shorter than trace)
        mag = np.sqrt(np.diff(f0) ** 2 + np.diff(f1) ** 2)
        # Radius of curvature (same length as trace)
        curv_r = _menger_curvature_radius(f0, f1, half_window=curv_hw)

        # Restrict search to is_touch = True epochs only
        touch_starts, touch_ends = (
            TrialTouchTimes.TouchEpoch & key & {'is_touch': True}
        ).fetch('start_time', 'end_time', order_by='epoch_idx')

        all_subm = []
        subm_idx = 0

        for t_start, t_end in zip(touch_starts, touch_ends):
            i0 = int(np.searchsorted(t, float(t_start)))
            i1 = int(np.searchsorted(t, float(t_end)))
            if i1 - i0 < min_dist:
                continue

            # mag is length n-1; curv_r is length n
            j0, j1 = i0, min(i1, len(mag))
            mag_ep  = mag[j0:j1]
            curv_ep = curv_r[i0:i1]

            speed_mins, _ = find_peaks(
                -mag_ep, distance=min_dist, prominence=params['speed_prominence']
            )
            curv_mins, _ = find_peaks(-curv_ep, distance=min_dist)

            # Co-occurring pairs → segment boundaries (epoch-local indices into mag_ep)
            boundaries = set()
            for sm in speed_mins:
                for cm in curv_mins:
                    if abs(int(sm) - int(cm)) <= cooc_win:
                        boundaries.add((int(sm) + int(cm)) // 2)
                        break
            boundaries = sorted(boundaries)

            seg_starts = [0] + [b + 1 for b in boundaries]
            seg_ends   = boundaries + [len(mag_ep) - 1]

            for s, e in zip(seg_starts, seg_ends):
                if e <= s:
                    continue
                abs_s = j0 + s
                abs_e = j0 + e
                if abs_s >= len(t) or abs_e >= len(t):
                    continue
                duration = float(t[abs_e] - t[abs_s])
                peak_spd = float(np.max(mag[abs_s:abs_e])) if abs_e > abs_s else 0.0
                all_subm.append({
                    **key,
                    'submovement_idx': subm_idx,
                    'start_time':      float(t[abs_s]),
                    'end_time':        float(t[abs_e]),
                    'duration':        duration,
                    'peak_speed':      peak_spd,
                })
                subm_idx += 1

        self.insert1({**key, 'n_submovements': subm_idx})
        self.Submovement.insert(all_subm)
