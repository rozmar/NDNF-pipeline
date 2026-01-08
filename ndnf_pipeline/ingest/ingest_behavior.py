#%
import harp
from .. import lab, experiment
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from PIL import Image
from zoneinfo import ZoneInfo
#%%
def ingest_behavior_sessions(dj):
    #%%
    df_surgery = pd.read_csv(dj.config['path.metadata']+'NDNF procedures_Surgeries.csv')
    df_surgery = df_surgery[1:] # skip explanation row
    subject_ids = df_surgery['Mouse ID'].tolist()
    metadata_files = os.listdir(dj.config['path.metadata'])
    for subject_id in subject_ids:
        if 'NDNF behavior notes_{}.csv'.format(subject_id) not in metadata_files:
            print('no behavior sessions for subject {}'.format(subject_id))
            continue
        df_subject = pd.read_csv(dj.config['path.metadata']+'NDNF behavior notes_{}.csv'.format(subject_id))
        for index, row in df_subject.iterrows():
            rig= row['Rig']
            experimenter = row['Experimenter']
            behavior_folder = os.path.join(dj.config['path.raw_data'],'behavior', rig,subject_id)
            session_date = datetime.strptime(row['Date'], '%Y/%m/%d').date() 
            session_folders_ = np.sort(os.listdir(behavior_folder))
            session_folder_dates = []
            session_folder_times = []
            session_folders = []
            for session_folder in session_folders_:
                if session_folder.startswith(subject_id+'_'):
                    if 'AIND' in rig:
                        session_datetime = datetime.strptime(session_folder[len(subject_id)+1:], '%Y%m%dT%H%M%S')
                        session_datetime = session_datetime.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("America/Los_Angeles"))
                    else:
                        print('what is the timezone of rig {}'.format(rig))
                        session_datetime = datetime.strptime(session_folder[len(subject_id)+1:], '%Y%m%dT%H%M%S')
                    session_folder_date = session_datetime.date()
                    session_folder_time = session_datetime.time()
                    session_folder_dates.append(session_folder_date)
                    session_folder_times.append(session_folder_time)
                    session_folders.append(session_folder)
            session_folder_dates = np.array(session_folder_dates)
            session_folder_times = np.array(session_folder_times)
            session_folders = np.array(session_folders)
            matching_indices = np.where(session_folder_dates == session_date)[0]
            session_time = np.min(session_folder_times[matching_indices])
            matching_sessions = experiment.Session()&{'subject_id':subject_id,'session_date':session_date,'session_time':session_time}  
            if len(matching_sessions)==0:
                session_id = len(experiment.Session()&{'subject_id':subject_id})
                session_dict = {'subject_id': subject_id,
                                'session': session_id,
                                'session_date': session_date,
                                'session_time': session_time,
                                'rig': rig,
                                'user_name': experimenter}
                session_details_dict = {'subject_id':subject_id,
                                        'session':session_dict['session'],
                                        'session_weight': row['Weight'],
                                        'session_water_earned': row['Water during experiment'],
                                        'session_water_extra': row['Extra water']}
                session_comment_dict = {'subject_id':subject_id,
                                        'session':session_dict['session'],
                                        'session_comment': row['Comment']}
                
                with dj.conn().transaction: #inserting the session
                    print('uploading session {}'.format(session_dict))            
                    experiment.Session().insert1(session_dict)
                    experiment.SessionDetails().insert1(session_details_dict)
                    experiment.SessionComment().insert1(session_comment_dict)
                    dj.conn().ping()
            else:
                session_dict = (experiment.Session()&{'subject_id':subject_id,'session_date':session_date,'session_time':session_time}  ).fetch1()
            matching_trials = experiment.SessionTrial()&session_dict
            if len(matching_trials)>0:
                print('trials already ingested for session {}, skipping'.format(session_dict))
                continue
            # get load cell calibration
            calibration_dates = (lab.Device.DeviceCalibration() &{'rig':rig,'device':'LoadCell'}).fetch('calibration_date')
            calibration_dates = calibration_dates[(calibration_dates-session_date)<timedelta(0)]
            calibration_date_needed = np.max(calibration_dates)
            calibration_dict = (lab.Device.DeviceCalibration() &{'rig':rig,'device':'LoadCell','calibration_date':calibration_date_needed}).fetch1('calibration_dict')
            p_to_g = []
            for loadcell_i in [0,1]:
                x = calibration_dict[loadcell_i]['g']
                y= calibration_dict[loadcell_i]['vals']
                p = np.polyfit(x,y,1)
                xx = [0,35]
                yy = np.polyval(p,xx)
                p_to_g.append(np.polyfit(y,x,1))

            task_settings_dict_list = []
            task_setting_dict_part_list = []
            sessiontrial_dict_list = []
            behaviortrial_dict_list = []
            forcetracedict_list = []
            forceaxis_dict_list = []
            trial_event_list = []
            reward_position_dict_list = []

            trials_so_far = 0
            for mi in matching_indices:
                session_folder = session_folders[mi]
                session_dir = os.path.join(behavior_folder,session_folder)
                with open(os.path.join(session_dir,'other/Config/tasklogic_output.json'),'r') as f:
                    tasklogic_dict = json.load(f)
                reward_size = tasklogic_dict['task_parameters']['environment']['block_statistics'][0]['trial_statistics']['left_harvest']['amount'] # TODO make this calibrated!!!
                valve_open_time = tasklogic_dict['task_parameters']['environment']['block_statistics'][0]['trial_statistics']['left_harvest']['amount']*0.05 # TODO make this calibrated!!!
                #loadcell_limit = tasklogic_dict['task_parameters']['operation_control']['force']['force_lookup_table']['left_max']# offset and scale is also here!!
                loadcell_limits_dict_ = {0:[tasklogic_dict['task_parameters']['operation_control']['force']['force_lookup_table']['left_min'],# TODO is it true that left corresponds to loadcell 0???
                                            tasklogic_dict['task_parameters']['operation_control']['force']['force_lookup_table']['left_max']],
                                        1:[tasklogic_dict['task_parameters']['operation_control']['force']['force_lookup_table']['right_min'],
                                            tasklogic_dict['task_parameters']['operation_control']['force']['force_lookup_table']['right_max']]}
                loadcell_offset = tasklogic_dict['task_parameters']['operation_control']['force']['force_lookup_table']['offset']# offset and scale is also here!!
                loadcell_scale = tasklogic_dict['task_parameters']['operation_control']['force']['force_lookup_table']['scale']# offset and scale is also here!!
                loadcell_limits = []
                loadcell_limits_dict = {}
                for pi,p in enumerate(p_to_g):
                    loadcell_limits_dict[pi] = np.polyval(p,loadcell_limits_dict_[pi])
                loadcell_extent = [loadcell_limits_dict[0][0],loadcell_limits_dict[0][1],loadcell_limits_dict[1][0],loadcell_limits_dict[1][1]]
                cached_threshold = tasklogic_dict['task_parameters']['environment']['block_statistics'][0]['trial_statistics']['left_harvest']['upper_force_threshold']
                force_duration = tasklogic_dict['task_parameters']['environment']['block_statistics'][0]['trial_statistics']['left_harvest']['force_duration']
                start_end_mm = tasklogic_dict['task_parameters']['environment']['block_statistics'][0]['trial_statistics']['left_harvest']['continuous_feedback']['converter_lut_output']
                distance = np.diff(start_end_mm)[0]
                forcemap_file = os.path.join(session_dir,"behavior/OperationControl/ForceLut.tiff")
            
                forcemap_im =Image.open(forcemap_file)
                forcemap_im  = np.asarray(forcemap_im,float)+loadcell_offset
                forcemap_im = forcemap_im/cached_threshold
                forcemap_im  = forcemap_im*distance # TODO make sure this is in mm/s!!!

                data_dict = {}
                for file in os.listdir(os.path.join(session_dir,'behavior/SoftwareEvents/')):
                    with open(os.path.join(session_dir,'behavior/SoftwareEvents/',file)) as f:
                        list_now = []
                        for line in f:
                            line = line.replace('null','None')
                            line = line.replace('false','False')
                            line = line.replace('true','True')
                            list_now.append(eval(line))
                        data_dict[file[:-5]] = list_now
                if 'TrialOutcome' not in data_dict.keys():
                    print('No TrialOutcome found in data_dict keys for session {},skipping'.format(session_dict))
                    continue
                hit =np.asarray([a['data']['HarvestAction']['action'] for a in data_dict['TrialOutcome']])=='Left'
                lickometer_data = harp.read(os.path.join(session_dir,'behavior/Lickometer.harp/LicketySplit_32.bin'))
                loadcell_data = harp.read(os.path.join(session_dir,"behavior/LoadCells.harp/LoadCells_33.bin"))
                csvfile = os.path.join(session_dir,'behavior/OperationControl/SpoutPosition.csv')
                lickport_df = pd.read_csv(csvfile)
                lickport_time = lickport_df['Seconds'].values
                lickport_position = lickport_df['Value'].values
                
                for pi,p in enumerate(p_to_g):
                    loadcell_data[pi] = np.polyval(p,loadcell_data[pi])
                # TODO : DO I NEED TO CORRECT FOR OFFSET HERE???
                #offset_0 = np.median(loadcell_data[0].values)
                #offset_1 = np.median(loadcell_data[1].values)
                
                loadcell_t = loadcell_data.index.values
                loadcell_0 = loadcell_data[0].values# - offset_0
                loadcell_1 = loadcell_data[1].values#  - offset_1
                loadcell_si = np.median(np.diff(loadcell_t))
                
                # define a few session-wide parameters
                task_protocol = ('DP',0) # ('DP', 0) is the only protocol so far. will extend, could be read from the google spreadsheet
                
                if len(task_settings_dict_list)==0:
                    task_setting_id = 0
                    task_settings_dict = {'subject_id':subject_id,
                                        'session':session_dict['session'],
                                        'task_setting_id': task_setting_id,
                                        'target_force_lut': forcemap_im,
                                        'reward_port_start_pos': start_end_mm[0],
                                        'reward_port_end_pos': start_end_mm[1],
                                        'reward_size': reward_size,
                                        }
                    task_settings_dict_list.append(task_settings_dict)
                    add_task_parts = True
                    
                else:
                    add_task_parts = False
                    new_task_settings = False
                    for tsd in task_settings_dict_list:
                        task_settings_dict = {'subject_id':subject_id,
                                            'session':session_dict['session'],
                                            'task_setting_id': tsd['task_setting_id'],
                                            'target_force_lut': forcemap_im,
                                            'reward_port_start_pos': start_end_mm[0],
                                            'reward_port_end_pos': start_end_mm[1],
                                            'reward_size': reward_size,
                                            }
                        same_dict = True
                        for k in tsd.keys():
                            if k=='target_force_lut':
                                if not np.array_equal(tsd[k],task_settings_dict[k]):
                                    same_dict = False
                                    break
                            else:
                                if tsd[k]!=task_settings_dict[k]:
                                    same_dict = False
                                    break
                        if same_dict:
                            new_task_settings = False
                            break
                    if new_task_settings:
                        task_setting_id = len(task_settings_dict_list)
                        task_settings_dict={'subject_id':subject_id,
                                                        'session':session_dict['session'],
                                                        'task_setting_id': task_setting_id,
                                                        'target_force_lut': forcemap_im,
                                                        'reward_port_start_pos': start_end_mm[0],
                                                        'reward_port_end_pos': start_end_mm[1],
                                                        'reward_size': reward_size,
                                                        }
                        task_settings_dict_list.append(task_settings_dict)
                        add_task_parts = True

                if add_task_parts:
                    for dim in [0,1]:
                        task_setting_dict_part = {'subject_id':subject_id,
                                                'session':session_dict['session'],
                                                'task_setting_id':task_setting_id,
                                                'force_axis_idx': dim,
                                                'target_force_axes': np.linspace(loadcell_limits_dict[dim][0],loadcell_limits_dict[dim][1],forcemap_im.shape[dim]),
                                                'force_direction': calibration_dict[dim]['direction'],
                                                }
                        task_setting_dict_part_list.append(task_setting_dict_part)
                
                reward_i = 0
                for trial_i, h in enumerate(hit): # go through trials
                    if trial_i==0:
                        session_zero_time = data_dict['Trial'][0]['timestamp']
                    trial_start_time = data_dict['Trial'][trial_i]['timestamp']
                    if tasklogic_dict['task_parameters']['environment']['block_statistics'][0]['trial_statistics']['response_period']['has_cue']:
                        go_cue_time = data_dict['ResponsePeriod'][trial_i]['timestamp']
                    else:
                        go_cue_time = np.nan
                    if trial_i == len(data_dict['TrialOutcome'])-1: #use this only for the last trial
                        trial_end_time = data_dict['TrialOutcome'][trial_i]['timestamp']
                    else:
                        trial_end_time = data_dict['Trial'][trial_i+1]['timestamp']
                    
                    if h:# data_dict['TrialOutcome'][trial_i]['data']['HarvestAction']['action'] == 'Left'
                        reward_timestamp = data_dict['HarvestActionSelected'][reward_i]['timestamp']
                        reward_i+=1
                        outcome = 'hit'
                    else:
                        reward_timestamp = np.nan
                        outcome = 'miss'
                    sessiontrial_dict = {'subject_id':subject_id,
                                        'session':session_dict['session'],
                                        'trial':trial_i+trials_so_far,
                                        'trial_start_time':trial_start_time-session_zero_time,
                                        'trial_end_time': trial_end_time-session_zero_time }
                    sessiontrial_dict_list.append(sessiontrial_dict)
                    behaviortrial_dict = {'subject_id': sessiontrial_dict['subject_id'],              
                                        'session':sessiontrial_dict['session'],
                                        'trial':sessiontrial_dict['trial'],
                                        'task': task_protocol[0],
                                        'task_protocol': task_protocol[1],
                                        'task_setting_id': task_settings_dict['task_setting_id'],
                                        'outcome':outcome}
                    behaviortrial_dict_list.append(behaviortrial_dict)
                    #%
                    # calculate lick lengths
                    lick_length = np.zeros(len(lickometer_data))*np.nan
                    for i in range(len(lickometer_data)):
                        if lickometer_data[0].values[i]==1:
                            lick_length[i] = lickometer_data.index.values[i+1]-lickometer_data.index.values[i]
                    lickometer_data['lick_length'] = lick_length
                    # get the licks for the trial from  lickometer_data and feed it to TrialEvent
                    licks_needed = (lickometer_data.index.values>=trial_start_time)&(lickometer_data.index.values<trial_end_time)
                    licks_now = lickometer_data[licks_needed]
                    licks_now = licks_now[licks_now[0].values==1]  # only licks, not no-licks
                    #%
                    trial_event_id = 0
                    if np.isfinite(go_cue_time):
                        go_dict = {'subject_id':subject_id,
                                    'session':session_dict['session'],
                                    'trial':trial_i+trials_so_far,
                                    'trial_event_type':'go',
                                    'trial_event_id':trial_event_id,
                                    'trial_event_time': go_cue_time - trial_start_time,
                                    'trial_event_duration': .1,# TODO HARD CODED TIME FOR GO CUE
                                        }
                        trial_event_id+=1
                        trial_event_list.append(go_dict)
                        
                    if h:
                        reward_dict = {'subject_id':subject_id,
                                    'session':session_dict['session'],
                                    'trial':trial_i+trials_so_far,
                                    'trial_event_type':'reward',
                                    'trial_event_id':trial_event_id,
                                    'trial_event_time': reward_timestamp - trial_start_time,
                                    'trial_event_duration': valve_open_time,
                                        }
                        trial_event_id+=1
                        trial_event_list.append(reward_dict)
                        

                    for lick_time, lick_row in licks_now.iterrows():
                        lick_dict = {'subject_id':subject_id,
                                    'session':session_dict['session'],
                                    'trial':trial_i+trials_so_far,
                                    'trial_event_type':'lick',
                                    'trial_event_id':trial_event_id,
                                    'trial_event_time': lick_time - trial_start_time,
                                    'trial_event_duration': lick_row['lick_length'],
                                    }
                        trial_event_id+=1
                        trial_event_list.append(lick_dict)

                    #%
                    # get the force traces and feed them to ForceTrace
                    force_needed = (loadcell_t>=trial_start_time)&(loadcell_t<trial_end_time)
                    force_now_t = loadcell_t[force_needed]-trial_start_time
                    force_now_0 = loadcell_0[force_needed]
                    force_now_1 = loadcell_1[force_needed]
                    forcetracedict = {'subject_id':subject_id,
                                        'session':session_dict['session'],
                                        'trial':trial_i+trials_so_far,
                                        'force_trace_time': force_now_t,
                                        }
                    forcetracedict_list.append(forcetracedict)
                    forceaxis_dict_0 = {'subject_id':subject_id,
                                        'session':session_dict['session'],
                                        'trial':trial_i+trials_so_far,
                                        'force_axis_idx': 0,
                                        'force_trace_value': force_now_0,
                                        }
                    forceaxis_dict_list.append(forceaxis_dict_0)
                    forceaxis_dict_1 = {'subject_id':subject_id,
                                        'session':session_dict['session'],
                                        'trial':trial_i+trials_so_far,
                                        'force_axis_idx': 1,
                                        'force_trace_value': force_now_1,
                                        }
                    forceaxis_dict_list.append(forceaxis_dict_1)
                    


                    #%
                    # get the lickport position and feed them to TrialRewardPortPosition

                    lickport_indices_needed = (lickport_time>=trial_start_time)&(lickport_time<trial_end_time)
                    lickport_now_t = lickport_time[lickport_indices_needed]-trial_start_time
                    lickport_now_pos = lickport_position[lickport_indices_needed]
                    reward_position_dict = {'subject_id':subject_id,
                                        'session':session_dict['session'],
                                        'trial':trial_i+trials_so_far,
                                        'reward_port_position_time': lickport_now_t,
                                        'reward_port_position_values': lickport_now_pos,
                                        }
                    reward_position_dict_list.append(reward_position_dict)
                    # and populate the schema


                    

                trials_so_far += trial_i+1 # for multiple files in a given session
            #% finally, do the insertion
            with dj.conn().transaction:
                print('uploading session {}'.format(session_dict))
                experiment.TaskSettings().insert(task_settings_dict_list)
                experiment.TaskSettings.ForceAxis().insert(task_setting_dict_part_list)
                experiment.SessionTrial().insert(sessiontrial_dict_list)
                experiment.BehaviorTrial().insert(behaviortrial_dict_list)
                experiment.TrialForceTrace().insert(forcetracedict_list)
                experiment.TrialForceTrace.TrialForceAxis().insert(forceaxis_dict_list)
                experiment.TrialEvent().insert(trial_event_list)
                experiment.TrialRewardPortPosition().insert(reward_position_dict_list)  
                dj.conn().ping()
                

