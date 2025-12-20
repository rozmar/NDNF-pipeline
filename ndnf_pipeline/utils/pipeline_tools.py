
#import logging

import datajoint as dj
#import hashlib
#log = logging.getLogger(__name__)


def get_schema_name(name):
    if name == 'lab':
        return 'pipeline_'+name
    elif dj.config['project'] == 'foraging':
        return 'group_shared_foraging-'+name
    elif dj.config['project'] == 'voltage imaging':
        return 'group_shared_voltageimaging-'+name
    elif dj.config['project'] == 'GENIE Calcium Imaging':
        return 'group_shared_geniecalciumimaging-'+name
    elif dj.config['project'] == 'bci-learning':
        return 'group_shared_bcilearning-'+name
    else:
        return None #

def drop_every_schema(schemaname):
    schema = dj.schema(schemaname+'_lab')
    schema.drop(force=True) 
 