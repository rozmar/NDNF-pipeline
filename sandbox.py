# #%% 0: DANGER !!! drop schemas
# from datetime import timedelta
# from ndnf_pipeline.utils.pipeline_tools import get_schema_name, drop_every_schema
# drop_every_schema('pipeline')
# #%% 
# schema = dj.schema('pipeline_analysis_log')#'pipeline_experiment')
# schema.drop(force=True) 

import datajoint as dj
dj.config.load('C:\\Secrets\\dj_local_conf.json')
dj.conn() 
direction_dict = {'LR':'Left - Right',
                  'RL':'Right - Left',
                  'AP':'Anterior - Posterior',
                  'PA':'Posterior - Anterior'}
#%% import necessary packages
import matplotlib.pyplot as plt
import pandas as pd
from ndnf_pipeline import lab, experiment, behavior_analysis
import numpy as np
# connect to the database

   #behavior_analysis.TrialTouchTimes.populate(display_progress=True)

   #behavior_analysis.TrialTouchTimes.delete
#ehavior_analysis.TrialTouchTimes.populate(display_progress=True)
#ehavior_analysis.TrialTiming.populate(display_progress=True)
#ehavior_analysis.BlockTiming.populate(display_progress=True)
experiment.TrialMetrics.populate(display_progress=True)



