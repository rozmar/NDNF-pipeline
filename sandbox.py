# #%% 0: DANGER !!! drop schemas
# from datetime import timedelta
# from ndnf_pipeline.utils.pipeline_tools import get_schema_name, drop_every_schema
# drop_every_schema('pipeline')
#%% 
# schema = dj.schema('pipeline_experiment')
# schema.drop(force=True) 

#%% 1: connect to datajoint using the local config file
import datajoint as dj
dj.config.load('/Users/neurozmar/Secrets/dj_local_conf_NDNF_behavior.json')
dj.conn()

#%% 2: update metadata from google drive
from ndnf_pipeline.utils.google_notebook import update_metadata
for metadata_spreadsheet in dj.config['metadata.spreadsheet_names']:
    update_metadata(metadata_spreadsheet, 
                    dj.config['path.metadata'], 
                    dj.config['path.google_creds_json'])
#%% 3: ingest metadata, behavior sessions
from ndnf_pipeline.ingest.ingest_metadata import ingest_metadata
from ndnf_pipeline.ingest.ingest_behavior import ingest_behavior_sessions
#ingest_metadata(dj)    
ingest_behavior_sessions(dj)


#%% behavior plotting
import datajoint as dj
dj.config.load('/Users/neurozmar/Secrets/dj_local_conf_NDNF_behavior.json')
dj.conn()
#%%
import matplotlib.pyplot as plt
import pandas as pd
from ndnf_pipeline import lab, experiment
#%%
subject_id = 'BCI81'
available_sessions = (experiment.Session()&{'subject_id':subject_id}).fetch('session')
trial_nums = []
session_lengths = []
for session in available_sessions:
    trials = (experiment.SessionTrial()&{'subject_id':subject_id,'session':session}).fetch('trial')
    trial_nums.append(len(trials))
    session_length = experiment.SessionTrial()&{'subject_id':subject_id,'session':session,'trial':np.max(trials)}.fetch('trial_end_time')
    session_lengths.append(session_length)




#%% PLOTTING - in progress
            
step_forward = int(t_forward/loadcell_si)
step_back= int(t_back/loadcell_si)

step_forward_reward = int(t_forward_reward/loadcell_si)
step_back_reward= int(t_back_reward/loadcell_si)

traces_0 = []
traces_1 = []
traces_0_reward = []
traces_1_reward = []
trace_time = np.arange(-1*step_back,step_forward)*loadcell_si
reward_trace_time = np.arange(-1*step_back_reward,step_forward_reward)*loadcell_si
for trial_i,(trial_start_t,reward_t) in enumerate(zip(trial_start_timestamp,reward_timestamp)):
    start_idx = np.argmax(loadcell_t>trial_start_t)
    reward_idx = np.argmax(loadcell_t>reward_t)
    if np.isnan(reward_t):
        traces_0.append(loadcell_0[start_idx-step_back:start_idx+step_forward])
        traces_1.append(loadcell_1[start_idx-step_back:start_idx+step_forward])
    else:
        reward_to_trial_idx = int((reward_t-trial_start_t)/loadcell_si)
        if reward_to_trial_idx<step_forward:
            trace_0 = np.zeros(step_back+step_forward)*np.nan
            trace_1 = np.zeros(step_back+step_forward)*np.nan
            trace_0[:step_back+reward_to_trial_idx] = loadcell_0[start_idx-step_back:start_idx+reward_to_trial_idx]
            trace_1[:step_back+reward_to_trial_idx] = loadcell_1[start_idx-step_back:start_idx+reward_to_trial_idx]
            traces_0.append(trace_0)
            traces_1.append(trace_1)
            
        else:
            traces_0.append(loadcell_0[start_idx-step_back:start_idx+step_forward])
            traces_1.append(loadcell_1[start_idx-step_back:start_idx+step_forward])
            
        if reward_to_trial_idx<step_back_reward:
            trace_0 = np.zeros(step_back_reward+step_forward_reward)*np.nan
            trace_1 = np.zeros(step_back_reward+step_forward_reward)*np.nan
            trace_0[step_back_reward-reward_to_trial_idx:] = loadcell_0[reward_idx-reward_to_trial_idx:reward_idx+step_forward_reward]
            trace_1[step_back_reward-reward_to_trial_idx:] = loadcell_1[reward_idx-reward_to_trial_idx:reward_idx+step_forward_reward]
            traces_0_reward.append(trace_0)
            traces_1_reward.append(trace_1)
            
        else:
            traces_0_reward.append(loadcell_0[reward_idx-step_back_reward:reward_idx+step_forward_reward])
            traces_1_reward.append(loadcell_1[reward_idx-step_back_reward:reward_idx+step_forward_reward])
            
            
traces_0 = np.asarray(traces_0)
traces_1 = np.asarray(traces_1)
traces_0_reward = np.asarray(traces_0_reward)
traces_1_reward = np.asarray(traces_1_reward)

#%plotting

trace_group_num = 20
trace_edges = np.arange(0,len(hit)+trace_group_num,trace_group_num)


fig = plt.figure(figsize = [20,8])
ax11 = fig.add_subplot(2,2,1)
ax11.set_title(session)
#ax1.plot(np.arange(len(hit_rate))[5:-5],hit_rate[5:-5],'k-')
ax11.set_xlabel('trial#')
#ax1.set_ylabel('hit rate (in 10 trials)')
ax1.set_title(session)
#ax11 = ax1.twinx()
ax11.plot(np.arange(len(time_to_reward)),time_to_reward,'g.')
ax11.plot(reward_trials[5:-5],time_to_reward_f[5:-5],'g-')
ax11.set_ylabel('time to get reward')
ax11.set_yscale('log')
licktimes = lickometer_data.index.values[lickometer_data[0].values == [1]]

ax_lick = fig.add_subplot(2,2,3,sharex = ax11)
for ti,rt in enumerate(reward_timestamp):
    if np.isnan(rt):
        ax_lick.plot(ti,0,'k.')
    else:
        
        licktimes_needed = (licktimes>rt-t_back_lick)&(licktimes<rt+t_forward_lick)
        ax_lick.plot(np.ones(sum(licktimes_needed))*ti,licktimes[licktimes_needed]-rt,'r_',markersize = 2)
    
ax_lick.set_xlabel('trial#')
ax_lick.set_ylabel('Time from reward (s)')

fig  = plt.figure(figsize = [15,25])
ax2 = fig.add_subplot(4,2,1)
ax2.set_title('around trial start')
for t,h in zip(traces_0,hit):
    ax2.plot(trace_time,t,'k-',alpha = .1)
ax2.plot(trace_time,np.nanmean(traces_0,0),'r-',alpha = 1,linewidth = 3)
if convert_to_g:
    ax2.set_ylabel('posterior-anterior (g)')
else:
    ax2.set_ylabel('posterior-anterior (au)')
ax2.set_xlabel('Time from trial start (s)')


ax3 = fig.add_subplot(4,2,3)
for t in traces_1:
    ax3.plot(trace_time,t,'k-',alpha = .1)
ax3.plot(trace_time,np.nanmean(traces_1,0),'r-',alpha = 1,linewidth = 3)
ax3.set_xlabel('Time from trial start (s)')
if convert_to_g:
    ax3.set_ylabel('latero-medial (g)')
else:
    ax3.set_ylabel('latero-medial (au)')
    

ax_traj = fig.add_subplot(4,2,4)

for t0,t1 in zip(traces_1,traces_0*-1): # yo WTF??? :D
    ax_traj.plot(t0,t1,'k-',alpha = .1)
if force_limits:
    ax_traj.set_xlim([-forced_limit,forced_limit])
    ax_traj.set_ylim([-forced_limit,forced_limit])
else:
    ax_traj.set_xlim([-loadcell_limit,loadcell_limit])
    ax_traj.set_ylim([-loadcell_limit,loadcell_limit])

if convert_to_g:
    ax_traj.set_xlabel('posterior-anterior (g)')
    ax_traj.set_ylabel('latero-medial (g)')
else:
    ax_traj.set_xlabel('posterior-anterior (au)')
    ax_traj.set_ylabel('latero-medial (au)')


## REWARD
try:
    ax2 = fig.add_subplot(4,2,5)
    ax2.set_title('around reward')
    for t in traces_0_reward:
        ax2.plot(reward_trace_time,t,'k-',alpha = .1)
    ax2.plot(reward_trace_time,np.nanmean(traces_0_reward,0),'r-',alpha = 1,linewidth = 3)
    if convert_to_g:
        ax2.set_ylabel('posterior-anterior (g)')
    else:
        ax2.set_ylabel('posterior-anterior (au)')
    ax2.set_xlabel('Time from reward (s)')
    
    
    ax3 = fig.add_subplot(4,2,7)
    for t in traces_1_reward:
        ax3.plot(reward_trace_time,t,'k-',alpha = .1)
    ax3.plot(reward_trace_time,np.nanmean(traces_1_reward,0),'r-',alpha = 1,linewidth = 3)
    ax3.set_xlabel('Time from reward (s)')
    if convert_to_g:
        ax3.set_ylabel('latero-medial (g)')
    else:
        ax3.set_ylabel('latero-medial (au)')
        
    
    ax_traj = fig.add_subplot(4,2,8)
    
    for t0,t1 in zip(traces_1_reward,traces_0_reward*-1): # yo WTF??? :D
        ax_traj.plot(t0,t1,'k-',alpha = .1)
    if force_limits:
        ax_traj.set_xlim([-forced_limit,forced_limit])
        ax_traj.set_ylim([-forced_limit,forced_limit])
    else:
        ax_traj.set_xlim([-loadcell_limit,loadcell_limit])
        ax_traj.set_ylim([-loadcell_limit,loadcell_limit])
    
    if convert_to_g:
        ax_traj.set_xlabel('posterior-anterior (g)')
        ax_traj.set_ylabel('latero-medial (g)')
    else:
        ax_traj.set_xlabel('posterior-anterior (au)')
        ax_traj.set_ylabel('latero-medial (au)')
except:
    pass

    
    
#%

fig = plt.figure(figsize = [12,12])
ax1 = fig.add_subplot(2,2,1)
ax1.set_title(session)
forcemap_im =Image.open(forcemap_file)
forcemap_im  = np.asarray(forcemap_im,float)+loadcell_offset
forcemap_im = forcemap_im/cached_threshold
forcemap_im  = forcemap_im*distance
if force_limits:
    im0 = plt.imshow(np.ones(forcemap_im.shape)*np.min(forcemap_im.flatten()),extent = [-forced_limit,forced_limit,-forced_limit,forced_limit],alpha = 1)
im = plt.imshow(forcemap_im,extent = [-loadcell_limit,loadcell_limit,-loadcell_limit,loadcell_limit],alpha = 1)
im0.set_clim(im.get_clim())
plt.colorbar(im,label = 'speed (mm/s)')
if force_limits:
    ax1.set_xlim([-forced_limit,forced_limit])
    ax1.set_ylim([-forced_limit,forced_limit])
if convert_to_g:
    ax1.set_xlabel('posterior-anterior (g)')
    ax1.set_ylabel('latero-medial (g)')
else:
    ax1.set_xlabel('posterior-anterior (au)')
    ax1.set_ylabel('latero-medial (au)')
ax2 = fig.add_subplot(2,2,2)
ax2.set_title('all data')
if convert_to_g:
    ax2.set_xlabel('posterior-anterior (g)')
    ax2.set_ylabel('latero-medial (g)')
else:
    ax2.set_xlabel('posterior-anterior (au)')
    ax2.set_ylabel('latero-medial (au)')
forcehist,binx,biny = np.histogram2d(loadcell_0.flatten(),loadcell_1.flatten(),histrange)
forcehist_all = forcehist.copy()
forcehist = np.log(forcehist/sum(forcehist.flatten()))
forcehist_ = forcehist.copy()
forcehist_ [np.isinf(forcehist )] = 0
forcehist[np.isinf(forcehist)] = np.nanmin(forcehist_.flatten())
im_hist = plt.imshow(forcehist,extent = extent,alpha = 1)
#
#im_hist .set_clim([0,.0005])
plt.colorbar(im_hist,label = 'fraction of time spent')

ax3 = fig.add_subplot(2,2,3)
if convert_to_g:
    ax3.set_xlabel('posterior-anterior (g)')
    ax3.set_ylabel('latero-medial (g)')
else:
    ax3.set_xlabel('posterior-anterior (au)')
    ax3.set_ylabel('latero-medial (au)')
ax3.set_title('Around trial start')
forcehist,binx,biny = np.histogram2d(traces_0.flatten(),traces_1.flatten(),histrange)
forcehist_trial_start = forcehist.copy()
forcehist = np.log(forcehist/sum(forcehist.flatten()))
forcehist_ = forcehist.copy()
forcehist_ [np.isinf(forcehist )] = 0
forcehist[np.isinf(forcehist)] = np.nanmin(forcehist_.flatten())
im_hist = plt.imshow(forcehist,extent = extent,alpha = 1)
#
#im_hist .set_clim([0,.0005])
plt.colorbar(im_hist,label = 'fraction of time spent')


ax3 = fig.add_subplot(2,2,4)
if convert_to_g:
    ax3.set_xlabel('posterior-anterior (g)')
    ax3.set_ylabel('latero-medial (g)') 
else:
    ax3.set_xlabel('posterior-anterior (au)')
    ax3.set_ylabel('latero-medial (au)')
ax3.set_title('Around hit')
forcehist,binx,biny = np.histogram2d(traces_0_reward.flatten(),traces_1_reward.flatten(),histrange)
forcehist_reward = forcehist.copy()
forcehist = np.log(forcehist/sum(forcehist.flatten()))
forcehist_ = forcehist.copy()
forcehist_ [np.isinf(forcehist )] = 0
forcehist[np.isinf(forcehist)] = np.nanmin(forcehist_.flatten())
im_hist = plt.imshow(forcehist,extent = extent,alpha = 1)
#
#im_hist .set_clim([0,.0005])
plt.colorbar(im_hist,label = 'fraction of time spent')



if len(forcehist_previous_reward)>0:
    #%
    
    fig = plt.figure()
    fig = plt.figure(figsize = [12,12])
    ax1 = fig.add_subplot(2,2,1)
    ax1.set_title('difference')
    forcemap_im =Image.open(forcemap_file)
    forcemap_im  = np.asarray(forcemap_im,float)+loadcell_offset
    forcemap_im = forcemap_im/cached_threshold
    forcemap_im  = forcemap_im*distance
    if force_limits:
        im0 = plt.imshow(np.ones(forcemap_im.shape)*np.min(forcemap_im.flatten()),extent = [-forced_limit,forced_limit,-forced_limit,forced_limit],alpha = 1)
    im = plt.imshow(forcemap_im,extent = [-loadcell_limit,loadcell_limit,-loadcell_limit,loadcell_limit],alpha = 1)
    plt.colorbar(im,label = 'speed')
    if force_limits:
        ax1.set_xlim([-forced_limit,forced_limit])
        ax1.set_ylim([-forced_limit,forced_limit])
    if convert_to_g:
        ax1.set_xlabel('posterior-anterior (g)')
        ax1.set_ylabel('latero-medial (g)')
    else:
        ax1.set_xlabel('posterior-anterior (au)')
        ax1.set_ylabel('latero-medial (au)')
    ax2 = fig.add_subplot(2,2,2)
    ax2.set_title('all data')
    if convert_to_g:
        ax2.set_xlabel('posterior-anterior (g)')
        ax2.set_ylabel('latero-medial (g)')
    else:
        ax2.set_xlabel('posterior-anterior (au)')
        ax2.set_ylabel('latero-medial (au)')
    a = forcehist_all/sum(forcehist_all.flatten())
    a[a==0] = np.nanmin(a[a>0])
    b = forcehist_previous_all/sum(forcehist_previous_all.flatten())
    b[b==0] = np.nanmin(b[b>0])
    toplot = np.log10(a/b)
    
    im_hist = plt.imshow(toplot ,extent = extent,alpha = 1)
    #im_hist.set_clim(clim_diff)
    
    #im_hist .set_clim([0,.0005])
    plt.colorbar(im_hist,label = 'fold-change time spent')
    ax3 = fig.add_subplot(2,2,3)
    if convert_to_g:
        ax3.set_xlabel('posterior-anterior (g)')
        ax3.set_ylabel('latero-medial (g)')
    else:
        ax3.set_xlabel('posterior-anterior (au)')
        ax3.set_ylabel('latero-medial (au)')
    ax3.set_title('Around trial start')
    
    a = forcehist_trial_start/sum(forcehist_trial_start.flatten())
    a[a==0] = np.nanmin(a[a>0])
    b = forcehist_previous_trial_start/sum(forcehist_previous_trial_start.flatten())
    b[b==0] = np.nanmin(b[b>0])
    toplot = np.log10(a/b)
    im_hist = plt.imshow(toplot ,extent = extent,alpha = 1)
    #im_hist.set_clim(clim_diff)
    
    #
    #
    #im_hist .set_clim([0,.0005])
    plt.colorbar(im_hist,label = 'fraction of time spent')

    ax3 = fig.add_subplot(2,2,4)
    if convert_to_g:
        ax3.set_xlabel('posterior-anterior (g)')
        ax3.set_ylabel('latero-medial (g)') 
    else:
        ax3.set_xlabel('posterior-anterior (au)')
        ax3.set_ylabel('latero-medial (au)')
    ax3.set_title('Around hit')
    
    try:
        a = forcehist_reward/sum(forcehist_reward.flatten())
        a[a==0] = np.nanmin(a[a>0])
        b = forcehist_previous_reward/sum(forcehist_previous_reward.flatten())
        b[b==0] = np.nanmin(b[b>0])
        toplot = np.log10(a/b)
        im_hist = plt.imshow(toplot,extent = extent,alpha = 1)
        #im_hist.set_clim(clim_diff)
        #
        #im_hist .set_clim([0,.0005])
        plt.colorbar(im_hist,label = 'fraction of time spent')
    except:
        pass
#fig = plt.figure()


forcehist_previous_reward = forcehist_reward.copy()
forcehist_previous_trial_start = forcehist_trial_start.copy()
forcehist_previous_all = forcehist_all.copy()
