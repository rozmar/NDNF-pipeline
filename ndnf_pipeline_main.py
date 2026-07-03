# this script downloads metadata, ingests data
#%% 1: connect to datajoint using the local config file
import datajoint as dj
dj.config.load('C:\\Secrets\\dj_local_conf.json')
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
from ndnf_pipeline import behavior_analysis,videography
#%%
ingest_metadata(dj)    
ingest_behavior_sessions(dj)
#%%
behavior_analysis.BlockStatistics.populate()
#videography.TrialVideo().populate(display_progress=True, reserve_jobs=True)
behavior_analysis.BlockStatistics.populate(display_progress=True, suppress_errors=True)
behavior_analysis.TrialSaturationTimes.populate(display_progress=True, suppress_errors=True)

behavior_analysis.TrialTouchTimes.populate(display_progress=True, suppress_errors=True)
behavior_analysis.TrialTiming.populate(display_progress=True, suppress_errors=True)
behavior_analysis.BlockTiming.populate(display_progress=True, suppress_errors=True)
#%%
behavior_analysis.schema.drop()

#%%
behavior_analysis.TrialTouchTimes.populate(display_progress=True, suppress_errors=True)
