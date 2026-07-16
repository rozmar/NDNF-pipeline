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
    """
    contents = [
        {'touch_param_id': 0, 'smooth_window_time': 0.3, 'sd_window_time': 0.05, 'sd_threshold': 0.05}
    ]


@schema
class TrialTouchTimes(dj.Computed):
    definition = """
    -> experiment.BehaviorTrial
    -> TouchDetectionParams
    ---
    n_touch_epochs: smallint  # number of contiguous touch epochs detected
    """

    @property
    def key_source(self):
        return experiment.BehaviorTrial * TouchDetectionParams & experiment.TrialForceTrace

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
        if len(force_trace_time)==0:
            print('empty trial')
            return
        sample_interval = float(np.median(np.diff(force_trace_time)))

        smooth_window = max(1, int(params['smooth_window_time'] / sample_interval))
        sd_window = max(2, int(params['sd_window_time'] / sample_interval))

        axis_indices, axis_values = (experiment.TrialForceTrace.TrialForceAxis & key).fetch(
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

        # Time array centered on each smoothing window
        half_w = smooth_window // 2
        smooth_time = force_trace_time[half_w: half_w + n_smooth]
        if len(smooth_time) < n_smooth:
            smooth_time = force_trace_time[0] + np.arange(n_smooth) * sample_interval

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
class BlockStatisticsTouch(dj.Computed):
    definition = """
    -> experiment.Block
    -> TouchDetectionParams
    ---
    touch_time_first_reward: float        # (s) total touch time in the first rewarded trial
    touch_time_last_rewards = null: float # (s) median total touch time over last 5 rewarded trials
    n_rewarded_trials: smallint
    """

    class ExpFit(dj.Part):
        definition = """
        -> master
        ---
        first_trial_value: float  # (s) fitted touch time at trial 0
        steady_state: float       # (s) asymptotic touch time
        time_constant: float      # (trials) exponential time constant tau
        r_squared: float
        """

    class InvFit(dj.Part):
        definition = """
        -> master
        ---
        first_trial_value: float  # (s) fitted touch time at trial 0
        steady_state: float       # (s) asymptotic touch time
        time_constant: float      # (trials) 1/x time constant
        r_squared: float
        """

    @property
    def key_source(self):
        return experiment.Block * TouchDetectionParams & TrialTouchTimes

    def make(self, key):
        rewarded_trials = (
            experiment.TrialEvent() * experiment.BehaviorTrial() * experiment.Block()
            & key & {'trial_event_type': 'threshold crossing'}
        ).fetch('trial', order_by='trial')

        if len(rewarded_trials) == 0:
            return

        touch_times = np.array([
            float((TrialTouchTimes.TouchEpoch() & key & {'trial': int(t), 'is_touch': True}
                   ).fetch('duration').astype(float).sum())
            for t in rewarded_trials
        ])

        first_trial = int((experiment.BehaviorTrial() & key).fetch('trial').min())
        trial_in_block = (rewarded_trials - first_trial).astype(float)

        touch_time_last = float(np.median(touch_times[-6:-1])) if len(touch_times) >= 6 else np.nan

        self.insert1({**key,
                      'touch_time_first_reward': float(touch_times[0]),
                      'touch_time_last_rewards': touch_time_last,
                      'n_rewarded_trials': len(rewarded_trials)})

        if len(rewarded_trials) < 4:
            return

        t = trial_in_block
        y = touch_times
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
