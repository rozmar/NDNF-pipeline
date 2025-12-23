# #%% drop schemas 
# from datetime import timedelta
# from ndnf_pipeline.utils.pipeline_tools import get_schema_name, drop_every_schema
# drop_every_schema('pipeline')

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

# %% ingest metadata
from ndnf_pipeline.ingest.ingest_metadata import ingest_metadata
ingest_metadata(dj)    
# %% Surgeries
from ndnf_pipeline import lab
import pandas as pd
from datetime import datetime
import numpy as np
#%% adding mouse lines


#%%
print('adding surgeries and stuff')
df_surgery = pd.read_csv(metadata_path+'NDNF procedures_Surgeries.csv')
for item in df_surgery.iterrows():
    item = item[1]
    if item['project'].lower() in ['behavior','marker gene hunt']:
        
        subjectdata = {
                'subject_id': item['animal#'],
                #'cage_number': item['cage#'],
                'date_of_birth': item['DOB'],
                'sex': item['sex'],
                'user_name': item['experimenter'],
                }
        
        try:
            lab.Subject().insert1(subjectdata)
        except dj.errors.DuplicateError:
            print('duplicate. animal :',item['animal#'], ' already exists')
        surgeryidx = 1
        while 'surgery date ('+str(surgeryidx)+')' in item.keys() and item['surgery date ('+str(surgeryidx)+')'] and type(item['surgery date ('+str(surgeryidx)+')']) == str:
            start_time = datetime.strptime(item['surgery date ('+str(surgeryidx)+')']+' '+item['surgery time ('+str(surgeryidx)+')'],'%Y-%m-%d %H:%M')
            end_time = start_time + timedelta(minutes = int(item['surgery length (min) ('+str(surgeryidx)+')']))
            surgerydata = {
                    'surgery_id': surgeryidx,
                    'subject_id':item['animal#'],
                    'user_name': item['experimenter'],
                    'start_time': start_time,
                    'end_time': end_time,
                    'surgery_description': item['surgery type ('+str(surgeryidx)+')'] + ':-: comments: ' + str(item[1]['surgery comments ('+str(surgeryidx)+')']),
                    }
            asdasdsa
            try:
                lab.Surgery().insert1(surgerydata)
            except dj.errors.DuplicateError:
                print('duplicate. surgery for animal ',item[1]['animal#'], ' already exists: ', start_time)
            #checking craniotomies
            #%
            cranioidx = 1
            while 'craniotomy diameter ('+str(cranioidx)+')' in item[1].keys() and item[1]['craniotomy diameter ('+str(cranioidx)+')'] and (type(item[1]['craniotomy surgery id ('+str(cranioidx)+')']) == int or type(item[1]['craniotomy surgery id ('+str(cranioidx)+')']) == float):
                if item[1]['craniotomy surgery id ('+str(cranioidx)+')'] == surgeryidx:
                    proceduredata = {
                            'surgery_id': surgeryidx,
                            'subject_id':item[1]['animal#'],
                            'procedure_id':cranioidx,
                            'skull_reference':item[1]['craniotomy reference ('+str(cranioidx)+')'],
                            'ml_location':item[1]['craniotomy lateral ('+str(cranioidx)+')'],
                            'ap_location':item[1]['craniotomy anterior ('+str(cranioidx)+')'],
                            'surgery_procedure_description': 'craniotomy: ' + str(item[1]['craniotomy comments ('+str(cranioidx)+')']),
                            }
                    try:
                        lab.Surgery.Procedure().insert1(proceduredata)
                    except dj.errors.DuplicateError:
                        print('duplicate cranio for animal ',item[1]['animal#'], ' already exists: ', cranioidx)
                cranioidx += 1
            #% 
            
            virusinjidx = 1
            while 'virus inj surgery id ('+str(virusinjidx)+')' in item[1].keys() and item[1]['virus inj virus id ('+str(virusinjidx)+')'] and item[1]['virus inj surgery id ('+str(virusinjidx)+')']:
                if item[1]['virus inj surgery id ('+str(virusinjidx)+')'] == surgeryidx:
# =============================================================================
#                     print('waiting')
#                     timer.sleep(1000)
# =============================================================================
                    if '[' in str(item[1]['virus inj lateral ('+str(virusinjidx)+')']):
                        virus_ml_locations = eval(item[1]['virus inj lateral ('+str(virusinjidx)+')'])
                    else:
                        virus_ml_locations = [int(item[1]['virus inj lateral ('+str(virusinjidx)+')'])]
                    if '[' in str(item[1]['virus inj anterior ('+str(virusinjidx)+')']):
                        virus_ap_locations = eval(item[1]['virus inj anterior ('+str(virusinjidx)+')'])
                    else:
                        virus_ap_locations = [int(item[1]['virus inj anterior ('+str(virusinjidx)+')'])]
                    if '[' in str(item[1]['virus inj ventral ('+str(virusinjidx)+')']):
                        virus_dv_locations = eval(item[1]['virus inj ventral ('+str(virusinjidx)+')'])
                    else:
                        virus_dv_locations = [int(item[1]['virus inj ventral ('+str(virusinjidx)+')'])]
                    if '[' in str(item[1]['virus inj volume (nl) ('+str(virusinjidx)+')']):
                        virus_volumes = eval(item[1]['virus inj volume (nl) ('+str(virusinjidx)+')'])
                    else:
                        virus_volumes = [int(item[1]['virus inj volume (nl) ('+str(virusinjidx)+')'])]
                    if '[' in str(item[1]['virus inj dilution ('+str(virusinjidx)+')']):
                        virus_dilutions = eval(item[1]['virus inj dilution ('+str(virusinjidx)+')'])
                    else:
                        virus_dilutions = np.ones(len(virus_ml_locations))*float(item[1]['virus inj dilution ('+str(virusinjidx)+')'])
                        
                        
                    for virus_ml_location,virus_ap_location,virus_dv_location,virus_volume,virus_dilution in zip(virus_ml_locations,virus_ap_locations,virus_dv_locations,virus_volumes,virus_dilutions):
                        injidx = len(lab.Surgery.VirusInjection() & surgerydata) +1
                        virusinjdata = {
                                'surgery_id': surgeryidx,
                                'subject_id':item[1]['animal#'],
                                'injection_id':injidx,
                                'virus_id':item[1]['virus inj virus id ('+str(virusinjidx)+')'],
                                'skull_reference':item[1]['virus inj reference ('+str(virusinjidx)+')'],
                                'ml_location':virus_ml_location,
                                'ap_location':virus_ap_location,
                                'dv_location':virus_dv_location,
                                'volume':virus_volume,
                                'dilution':virus_dilution, #item[1]['virus inj dilution ('+str(virusinjidx)+')'],
                                'description': 'virus injection: ' + str(item[1]['virus inj comments ('+str(virusinjidx)+')']),
                                }
                        try:
                            lab.Surgery.VirusInjection().insert1(virusinjdata)
                        except dj.errors.DuplicateError:
                            print('duplicate virus injection for animal ',item[1]['animal#'], ' already exists: ', injidx)
                virusinjidx += 1    
            #%
            
            surgeryidx += 1
                
