"""Figure 03 of M033_deletion_sessions_analysis.ipynb ('post-arrival pull'), for M032 sessions 7 and 8, block 1."""
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import datajoint as dj
dj.config.load(str(repo_root / 'dj_local_conf.json'))
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from ndnf_pipeline import experiment
from ndnf_pipeline.plot import behavior_plots as bp

subject_id = 'M032'
SESSIONS = [7, 8]
BLOCK = 1
FILTER_SIGMA_MS = 3.0
DOWNSAMPLE = 5
FS = 1000 / DOWNSAMPLE
ACTIVE_G = 3.0
SESSION_COLORS = {7: '#5a5a5a', 8: '#e34a33'}

fig_dir = repo_root / 'Notebooks' / 'figures' / subject_id
fig_dir.mkdir(parents=True, exist_ok=True)


def style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def lickport_arrival(t, v, tol_mm=0.05):
    t, v = np.asarray(t, float), np.asarray(v, float)
    if len(v) == 0:
        return np.nan
    at_end = v >= v.max() - tol_mm
    i = len(v) - 1
    while i > 0 and at_end[i - 1]:
        i -= 1
    return t[i]


def load_block(key):
    tsid = (experiment.TaskSettings() * experiment.Block() & key).fetch1('task_setting_id')
    _, lr_idx, pa_idx, lr_sign, pa_sign = bp._get_block_force_axes(experiment, key['subject_id'], key['session'], tsid)
    axes = (lr_idx, pa_idx, lr_sign, pa_sign)
    lp_trials, lp_t, lp_v = (experiment.TrialRewardPortPosition() * experiment.BehaviorTrial() & key).fetch(
        'trial', 'reward_port_position_time', 'reward_port_position_values')
    arrival = {int(tr): lickport_arrival(t, v) for tr, t, v in zip(lp_trials, lp_t, lp_v)}
    trials, times = (experiment.TrialForceTrace() * experiment.BehaviorTrial() & key).fetch(
        'trial', 'force_trace_time', order_by='trial')
    tr0, f0 = (experiment.TrialForceTrace.TrialForceAxis() * experiment.BehaviorTrial() & key
               & {'force_axis_idx': 0}).fetch('trial', 'force_trace_value', order_by='trial')
    tr1, f1 = (experiment.TrialForceTrace.TrialForceAxis() * experiment.BehaviorTrial() & key
               & {'force_axis_idx': 1}).fetch('trial', 'force_trace_value', order_by='trial')
    assert np.array_equal(trials, tr0) and np.array_equal(trials, tr1)
    b0, b1 = np.median(np.concatenate(f0)), np.median(np.concatenate(f1))
    lr_all, pa_all = bp._lr_pa_traces([f - b0 for f in f0], [f - b1 for f in f1], *axes)

    go = bp._event_time_by_trial(experiment, key, 'go')
    thr = bp._event_time_by_trial(experiment, key, 'threshold crossing')
    hits = set((experiment.BehaviorTrial() & key & {'outcome': 'hit'}).fetch('trial').astype(int))
    out = []
    for trial, t, lr, pa in zip(trials, times, lr_all, pa_all):
        trial = int(trial)
        if trial not in hits or trial not in go or trial not in thr:
            continue
        t = np.asarray(t, float)
        sigma = FILTER_SIGMA_MS / 1000 / np.median(np.diff(t))
        lr = gaussian_filter1d(np.asarray(lr, float), sigma)
        pa = gaussian_filter1d(np.asarray(pa, float), sigma)
        m = (t >= go[trial]) & (t < thr[trial])
        lr_d, pa_d = lr[m][::DOWNSAMPLE], pa[m][::DOWNSAMPLE]
        arr = arrival.get(trial, np.nan)
        arr = thr[trial] - 0.5 if not np.isfinite(arr) else arr
        n_pre = int(np.clip(round((arr - go[trial]) * FS), 1, len(lr_d)))
        out.append(dict(session=key['session'], trial=trial, arrival_lag=thr[trial] - arr,
                        lr=lr_d, pa=pa_d, n_pre=n_pre))
    return out


def pull_samples(lr, pa):
    """Active samples (|F| >= 3 g) pointing within 45 deg of straight back (pull)."""
    bear = np.degrees(np.arctan2(lr, -pa))
    return int(((np.hypot(lr, pa) >= ACTIVE_G) & (np.abs(bear) <= 45)).sum())


resp = []
for s in SESSIONS:
    tr = load_block({'subject_id': subject_id, 'session': s, 'block': BLOCK})
    print(f's{s} b{BLOCK}: {len(tr)} rewarded trials')
    resp.extend(tr)

rng = np.random.default_rng(0)
win = int(2.0 * FS)
t_al = np.arange(-win, 0) / FS
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), gridspec_kw=dict(width_ratios=[0.8, 1.2, 0.8]))

ax = axes[0]
for s in SESSIONS:
    lags = np.array([r['arrival_lag'] for r in resp if r['session'] == s])
    ax.scatter(s + rng.uniform(-0.25, 0.25, len(lags)), lags, s=8, color=SESSION_COLORS[s], alpha=0.5, lw=0)
    ax.plot(s, np.median(lags), 'o', color='k', ms=6)
ax.plot(SESSIONS, [np.median([r['arrival_lag'] for r in resp if r['session'] == s]) for s in SESSIONS], '-', color='k', lw=1)
ax.set_xticks(SESSIONS); ax.set_xlim(SESSIONS[0] - 0.7, SESSIONS[-1] + 0.7)
ax.set_ylim(0, 1.2); ax.set_ylabel('threshold crossing -\nlickport arrival (s)'); ax.set_xlabel('session')
ax.set_title('(a) arrival precedes the threshold event'); style(ax)

ax = axes[1]
for s in SESSIONS:
    R = [r for r in resp if r['session'] == s and len(r['pa']) >= win]
    M = np.vstack([r['pa'][-win:] for r in R])
    mu, se = M.mean(0), M.std(0) / np.sqrt(len(M))
    ax.plot(t_al, mu, color=SESSION_COLORS[s], lw=2, label=f's{s} b{BLOCK} (n={len(M)})')
    ax.fill_between(t_al, mu - se, mu + se, color=SESSION_COLORS[s], alpha=0.2, lw=0)
med_lag = np.median([r['arrival_lag'] for r in resp])
ax.axvline(-med_lag, color='0.3', ls=':', lw=1.2)
ax.text(-med_lag, 1.0, ' lickport arrives', transform=ax.get_xaxis_transform(), fontsize=8, va='top')
ax.axhline(0, color='0.7', lw=0.8)
ax.set_xlabel('time from threshold crossing (s)'); ax.set_ylabel('Posterior (pull) - Anterior (push) (g)')
ax.set_title('(b) force around reward'); ax.legend(frameon=False, fontsize=8, loc='lower left'); style(ax)

ax = axes[2]
shares = []
for s in SESSIONS:
    R = [r for r in resp if r['session'] == s]
    n_pre = sum(pull_samples(r['lr'][:r['n_pre']], r['pa'][:r['n_pre']]) for r in R)
    n_post = sum(pull_samples(r['lr'][r['n_pre']:], r['pa'][r['n_pre']:]) for r in R)
    shares.append(n_post / (n_pre + n_post))
    print(f's{s}: post-arrival share of pull time = {shares[-1]:.3f}')
ax.bar(SESSIONS, shares, color=[SESSION_COLORS[s] for s in SESSIONS], width=0.6)
ax.set_xticks(SESSIONS); ax.set_xlim(SESSIONS[0] - 0.7, SESSIONS[-1] + 0.7)
ax.set_ylim(0, 1); ax.set_ylabel('share of response-epoch\npull time after arrival'); ax.set_xlabel('session')
ax.set_title('(c) how much of the pulling is post-arrival'); style(ax)

fig.suptitle(f'{subject_id}: sessions {SESSIONS[0]} and {SESSIONS[1]}, block {BLOCK}', y=1.02)
fig.tight_layout()
out = fig_dir / f'{subject_id}_03_post_arrival_pull_s7_s8_b{BLOCK}.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
print('saved', out)
