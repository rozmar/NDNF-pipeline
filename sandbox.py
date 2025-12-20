#%% drop schemas 
from ndnf_pipeline.utils.pipeline_tools import get_schema_name, drop_every_schema
drop_every_schema('pipeline')

#%% connect to datajoint
import datajoint as dj
dj.config.load('/Users/neurozmar/Secrets/dj_local_conf_NDNF_behavior.json')
dj.conn()

# %% update metadata from google drive
from ndnf_pipeline.utils.google_notebook import update_metadata

for metadata_spreadsheet in dj.config['metadata.spreadsheet_names']:
    update_metadata(metadata_spreadsheet, 
                    dj.config['path.metadata'], 
                    dj.config['path.google_creds_json'])
    
#%% experimenters
import pandas as pd
import pipeline.lab as lab

metadata_path = dj.config['path.metadata']
df_experimenters = pd.read_csv(metadata_path+'NDNF experimenters_People.csv')
experimenterdata = list()
for experimenter in df_experimenters.iterrows():
    experimenter = experimenter[1]
    dictnow = {'user_name':experimenter['ID'],'full_name':'{} {}'.format(experimenter['First Name'],experimenter['Last Name'])}
    experimenterdata.append(dictnow)
print('adding experimenters') 
for experimenternow in experimenterdata:
    try:
        lab.Person().insert1(experimenternow)
    except dj.errors.DuplicateError:
        print('duplicate. experimenter: ',experimenternow['username'], ' already exists')

#%% rigs
df_rigs = pd.read_csv(metadata_path+'NDNF experimenters_Rigs.csv')
rigdata = list()
rigstatedata = list()
for rig in df_rigs.iterrows():
    rig = rig[1]
    dictnow = {'rig':rig['Rig ID'],
               'rig_update_date':rig['Date'],
               'building':rig['Building'],
               'room':rig['Room'],
               'rig_description':rig['Description']}
    rigstatedata.append(dictnow)
    rigdata.append({'rig':rig['Rig ID']})

print('adding rigs')
for rignow in rigdata:
    try:
        lab.Rig.insert1(rignow)
    except dj.errors.DuplicateError:
        print('duplicate. rig: ',rignow['rig'], ' already exists')
for rignow in rigstatedata:
    try:
        lab.Rig.RigState().insert1(rignow)
    except dj.errors.DuplicateError:
        print('duplicate rigstate: ',rignow['rig'], ' already exists')


#% -  do the rest from here:
# https://github.com/rozmar/BCI_pipeline/blob/main/pipeline/ingest/datapipeline_metadata.py
#add subjects
