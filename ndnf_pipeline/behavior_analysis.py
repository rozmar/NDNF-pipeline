import datajoint as dj
import numpy as np
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
