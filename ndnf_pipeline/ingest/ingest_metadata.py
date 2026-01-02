#%% experimenters
import pandas as pd
#import ndnf_pipeline.lab as lab
from .. import lab
from datetime import datetime
from datetime import timedelta
import numpy as np
def ingest_metadata(dj):
    ingest_experimenters(dj)
    ingest_rigs(dj)
    ingest_devices_and_calibrations(dj)
    ingest_viruses(dj)
    ingest_mouse_lines(dj)
    ingest_surgeries(dj)

    # TODO: ingest devices and calibrations

def ingest_experimenters(dj):
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
            print('duplicate. experimenter: ',experimenternow['user_name'], ' already exists')

def ingest_rigs(dj):
    metadata_path = dj.config['path.metadata']
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
        rigdata.append({'rig':rig['Rig ID'],
                        'institute_id': rig['Institute']})

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


def ingest_viruses(dj):
    metadata_path = dj.config['path.metadata']
    df_virus_stocks = pd.read_csv(metadata_path+'NDNF viruses plasmids_Virus stocks.csv')

    virusdata = list()
    serotypedata = list()
    for virus in df_virus_stocks.iterrows():
        virus = virus[1]
        if type(virus['Remarks']) != str:
            virus['Remarks'] = ''
        dictnow = {'virus_id':virus['Virus ID'],
                    'virus_source':virus['Origin'],
                    'serotype':virus['Virus Serotype'],
                    'user_name':virus['Name'],
                    'virus_name':virus['Virus Name'],
                    'titer':virus['Titer'],
                    'order_date':virus['Production Date'],
                    'addgene_id':virus['Addgene#'],
                    'lot_number':virus['Order#/LOT#'],
                    'virus_remarks':virus['Remarks']}
        virusdata.append(dictnow)
        dictnow = {'serotype':virus['Virus Serotype']}

        serotypedata.append(dictnow)
    #%
    print('adding Viruses and Serotypes')
    for virusnow,serotypenow in zip(virusdata,serotypedata):
        try:
            lab.Serotype().insert1(serotypenow)
        except dj.errors.DuplicateError:
            print('duplicate serotype: ',serotypenow['serotype'], ' already exists')
        try:
            lab.Virus().insert1(virusnow)
        except dj.errors.DuplicateError:
            print('duplicate virus: ',virusnow['virus_name'], ' already exists')
    #%% add solutions and virus aliquots
    df_virus_solutions = pd.read_csv(metadata_path+'NDNF viruses plasmids_Solutions.csv')
    solutiondata = list()
    solutioncomponentdata = list()
    for solution in df_virus_solutions.iterrows():
        solution = solution[1]
        if type(solution['Remarks']) != str:
            solution['Remarks'] = ''
        
        dictnow = {'solution_id':solution['Solution ID'],
                'user_name':solution['Name'],
                    'prep_date':solution['Prep date'],
                    'solution_remarks':solution['Remarks']}
        solutiondata.append(dictnow)
        ingredient_counter = 1
        while 'Ingredient_{}'.format(ingredient_counter) in solution.keys():
            ingredient = solution['Ingredient_{}'.format(ingredient_counter)]
            amount = solution['Amount_{}'.format(ingredient_counter)]
            if type(ingredient) == str and type(amount) == str:
                dictcomponentnow = {'solution_id':solution['Solution ID'],
                                    'ingredient':ingredient,
                                    'amount':amount}
                solutioncomponentdata.append(dictcomponentnow)
            ingredient_counter += 1
    print('adding solutions')
    for solutionnow in solutiondata:
        try:
            lab.Solution().insert1(solutionnow)
        except dj.errors.DuplicateError:
            print('duplicate solution: ',solutionnow['solution_id'], ' already exists')
    for solutioncomponentnow in solutioncomponentdata:
        try:
            lab.Solution.SolutionComponent().insert1(solutioncomponentnow)
        except dj.errors.DuplicateError:
            print('duplicate solution component for solution: ',solutioncomponentnow['solution_id'], ' already exists')

    #%% add virus aliquots
    df_virus_aliquots = pd.read_csv(metadata_path+'NDNF viruses plasmids_Virus aliquots.csv')
    virusaliquotdata = list()
    for virusaliquot in df_virus_aliquots.iterrows():
        virusaliquot = virusaliquot[1]
        if type(virusaliquot['Remarks']) != str:
            virusaliquot['Remarks'] = ''
        
        dictnow = {'virus_id':virusaliquot['Origin ID'],
                    'virus_aliquot_id':virusaliquot['Aliquot ID'],
                    'dilution':virusaliquot['Dilution factor'],
                    'prep_date':virusaliquot['Dilution date'],
                    'virus_aliquot_remarks':virusaliquot['Remarks']}
        if type(virusaliquot['Solution used']) == str:
            dictnow['solution_id'] = virusaliquot['Solution used']
        else:
            dictnow['solution_id'] = None
        if dictnow['solution_id'].strip(' ') == 'none' or dictnow['solution_id'].strip(' ')=='-':
            dictnow['solution_id'] = None
        virusaliquotdata.append(dictnow)
    for virusaliquotnow in virusaliquotdata:
        try:
            lab.VirusAliquot().insert1(virusaliquotnow)
        except dj.errors.DuplicateError:
            print('duplicate virus aliquot: ',virusaliquotnow['virus_aliquot_id'], ' already exists')

    #%%% add virus mixtures
    df_virus_mixtures = pd.read_csv(metadata_path+'NDNF viruses plasmids_Virus mixtures.csv')
    virusmixturedata = list()
    virusmixturepartdata = list()
    for virusmixture in df_virus_mixtures.iterrows():
        virusmixture = virusmixture[1]
        if type(virusmixture['Remarks']) != str:
            virusmixture['Remarks'] = ''
        
        dictnow = {'virus_mixture_id':virusmixture['Mixture ID'],
                    'prep_date':virusmixture['Mixture date'],
                    'virus_mixture_remarks':virusmixture['Remarks']}
        virusmixturedata.append(dictnow)
        part_counter = 1
        while 'Origin ID {}'.format(part_counter) in virusmixture.keys():
            virus_aliquot = virusmixture['Origin ID {}'.format(part_counter)]
            
            if type(virus_aliquot) == str:
                virus_aliquot_dict = (lab.VirusAliquot & {'virus_aliquot_id': virus_aliquot}).fetch1()
                dictpartnow = {'virus_mixture_id':virusmixture['Mixture ID'],
                            'virus_id':virus_aliquot_dict['virus_id'],
                                'virus_aliquot_id':virus_aliquot_dict['virus_aliquot_id'],
                                'fraction':virusmixture['Fraction {}'.format(part_counter)]}
                virusmixturepartdata.append(dictpartnow)
            part_counter += 1
    for virusmixturenow in virusmixturedata:
        try:
            lab.VirusMixture().insert1(virusmixturenow)
        except dj.errors.DuplicateError:
            print('duplicate virus mixture: ',virusmixturenow['virus_mixture_id'], ' already exists')
    for virusmixturepartnow in virusmixturepartdata:
        try:
            lab.VirusMixture.VirusMixturePart().insert1(virusmixturepartnow)
        except dj.errors.DuplicateError:
            print('duplicate virus mixture part for mixture: ',virusmixturepartnow['virus_mixture_id'], ' already exists')

def ingest_mouse_lines(dj):
    df_mouselines = pd.read_csv(dj.config['path.metadata']+'NDNF animals_Mouse lines.csv')
    for mouseline in df_mouselines.iterrows():
        mouseline = mouseline[1]
        mouselinedata = {
                'mouse_line': mouseline['Mouse Line ID'],
                'line_description': mouseline['Mouse Line Description'],
                'jax_stock_number': mouseline['JAX#'] if not pd.isna(mouseline['JAX#']) else None,
                }
        try:
            lab.MouseLine().insert1(mouselinedata)
        except dj.errors.DuplicateError:
            print('duplicate. mouse line :',mouseline['Mouse Line ID'], ' already exists')
#
def ingest_surgeries(dj):
    print('adding surgeries and stuff')
    df_surgery = pd.read_csv(dj.config['path.metadata']+'NDNF procedures_Surgeries.csv')
    df_surgery = df_surgery[1:] # skip explanation row
    for item in df_surgery.iterrows():
        item = item[1]
        if item['project'].lower() not in ['behavior','marker gene hunt']:
            continue
        subjectdata = {
                'subject_id': item['Mouse ID'],
                'institutional_id': item['animal#'],
                'institute_id': item['Institute'],
                #'cage_number': item['cage#'],
                'date_of_birth': item['DOB'],
                'sex': item['sex'],
                'user_name': item['experimenter'],
                }
        lineage_list = []
        genotypeidx = 1
        while 'genotype_'+str(genotypeidx) in item.keys():
            if type(item['genotype_'+str(genotypeidx)]) == str and item['genotype_'+str(genotypeidx)] != '' and item['genotype_'+str(genotypeidx)] != '-':
                lineage_list.append( {'subject_id':subjectdata['subject_id'],
                                    'mouse_line':item['genotype_'+str(genotypeidx)], 
                                    'zygosity':'Unknown'} )
            genotypeidx += 1
        try:
            lab.Subject().insert1(subjectdata)
        except dj.errors.DuplicateError:
            print('duplicate. animal :',item['animal#'], ' already exists')
        for lineage in lineage_list:
            try:
                lab.Subject.Lineage().insert1(lineage)
            except dj.errors.DuplicateError:
                print('duplicate lineage. animal :',item['animal#'], ' mouse line: ',lineage['mouse_line'], ' already exists')
        surgeryidx = 1
        while 'surgery date ('+str(surgeryidx)+')' in item.keys() and item['surgery date ('+str(surgeryidx)+')'] and type(item['surgery date ('+str(surgeryidx)+')']) == str:
            start_time = datetime.strptime(item['surgery date ('+str(surgeryidx)+')']+' '+item['surgery time ('+str(surgeryidx)+')'],'%Y/%m/%d %H:%M')
            end_time = start_time + timedelta(minutes = int(item['surgery length (min) ('+str(surgeryidx)+')']))
            surgerydata = {
                'surgery_id': surgeryidx,
                'subject_id':item['animal#'],
                'user_name': item['experimenter'],
                'date':start_time.date(),
                'start_time': start_time,
                'end_time': end_time,
                'surgery_description': item['surgery type ('+str(surgeryidx)+')'],
                'surgery_notes':str(item['surgery comments ('+str(surgeryidx)+')'])
                }
            
            try:
                lab.Surgery().insert1(surgerydata)
            except dj.errors.DuplicateError:
                print('duplicate. surgery for animal ',item['animal#'], ' already exists: ', start_time)
            #checking craniotomies
            #%

            cranioidx = 1
            while 'craniotomy diameter ('+str(cranioidx)+')' in item.keys() and item['craniotomy diameter ('+str(cranioidx)+')'] and (type(item['craniotomy surgery id ('+str(cranioidx)+')']) == int or type(item['craniotomy surgery id ('+str(cranioidx)+')']) == float):
                if item['craniotomy surgery id ('+str(cranioidx)+')'] == surgeryidx:
                    craniotomydata = { # this is not right here
                            'surgery_id': surgeryidx,
                            'subject_id':item['animal#'],
                            'procedure_id':cranioidx,
                            'skull_reference':item['craniotomy reference ('+str(cranioidx)+')'],
                            'ml_location':item['craniotomy lateral ('+str(cranioidx)+')'],
                            'ap_location':item['craniotomy anterior ('+str(cranioidx)+')'],
                            'surgery_procedure_description': 'craniotomy: ' + str(item['craniotomy comments ('+str(cranioidx)+')']),
                            }
                    try:
                        lab.Surgery.Craniotomy().insert1(craniotomydata)
                    except dj.errors.DuplicateError:
                        print('duplicate cranio for animal ',item['animal#'], ' already exists: ', cranioidx)
                cranioidx += 1
            virusinjidx = 1
            while 'virus inj surgery id ('+str(virusinjidx)+')' in item.keys() and item['virus inj virus id ('+str(virusinjidx)+')'] and item['virus inj surgery id ('+str(virusinjidx)+')']:
                if item['virus inj surgery id ('+str(virusinjidx)+')'] == surgeryidx:
                    injection_num = []
                    if '[' in str(item['virus inj ML coordinate ('+str(virusinjidx)+')']):
                        virus_ml_locations = eval(item['virus inj ML coordinate ('+str(virusinjidx)+')'])
                        injection_num.append(len(virus_ml_locations))
                    else:
                        virus_ml_locations = [int(item['virus inj ML coordinate ('+str(virusinjidx)+')'])]
                    if '[' in str(item['virus inj AP coordinate ('+str(virusinjidx)+')']):
                        virus_ap_locations = eval(item['virus inj AP coordinate ('+str(virusinjidx)+')'])
                        injection_num.append(len(virus_ap_locations))
                    else:
                        virus_ap_locations = [int(item['virus inj AP coordinate ('+str(virusinjidx)+')'])]
                    if '[' in str(item['virus inj DV coordinate ('+str(virusinjidx)+')']):
                        virus_dv_locations = eval(item['virus inj DV coordinate ('+str(virusinjidx)+')'])
                        injection_num.append(len(virus_dv_locations))
                    else:
                        virus_dv_locations = [int(item['virus inj DV coordinate ('+str(virusinjidx)+')'])]
                    if '[' in str(item['virus inj volume (nl) ('+str(virusinjidx)+')']):
                        virus_volumes = eval(item['virus inj volume (nl) ('+str(virusinjidx)+')'])
                        injection_num.append(len(virus_volumes))
                    else:
                        virus_volumes = [int(item['virus inj volume (nl) ('+str(virusinjidx)+')'])]
                    if '[' in item['virus inj virus id ('+str(virusinjidx)+')']:
                        virus_ids = eval(item['virus inj virus id ('+str(virusinjidx)+')'])
                        injection_num.append(len(virus_ids))
                    else:
                        virus_ids = [item['virus inj virus id ('+str(virusinjidx)+')']]
                    burrhole = item['virus inj burrhole (T/F) ('+str(virusinjidx)+')']
                    
                    if len(np.unique(injection_num)) > 1:
                        raise ValueError('Mismatch in number of virus injection parameters for animal ', item['animal#'], ' surgery ', surgeryidx, ' injection ', virusinjidx)
                    else:
                        injection_num = injection_num[0]

                    if len(virus_ml_locations) != injection_num:
                        virus_ml_locations = virus_ml_locations * injection_num
                    if len(virus_ap_locations) != injection_num:
                        virus_ap_locations = virus_ap_locations * injection_num
                    if len(virus_dv_locations) != injection_num:
                        virus_dv_locations = virus_dv_locations * injection_num
                    if len(virus_volumes) != injection_num:     
                        virus_volumes = virus_volumes * injection_num
                    if len(virus_ids) != injection_num:
                        virus_ids = virus_ids * injection_num   

                    virus_ap_ml_reference = item['virus inj APML reference ('+str(virusinjidx)+')']
                    virus_dv_reference = item['virus inj DV reference ('+str(virusinjidx)+')']
                    if virus_dv_reference in ['dura','Dura','DURA','surface of the brain','Surface of the brain']:
                        virus_dv_reference = 'dura'
                    virus_ap_direction = item['virus inj AP direction ('+str(virusinjidx)+')']
                    virus_ml_direction = item['virus inj ML direction ('+str(virusinjidx)+')']
                    virus_dv_direction = item['virus inj DV direction ('+str(virusinjidx)+')']
                    
                    for virus_ml_location,virus_ap_location,virus_dv_location,virus_volume,virus_id in zip(virus_ml_locations,virus_ap_locations,virus_dv_locations,virus_volumes,virus_ids):
                        injidx = len(lab.Surgery.VirusInjection() & surgerydata) +1
                        coordinatesystemdata =  {'ap_ml_reference':virus_ap_ml_reference,
                                            'dv_reference':virus_dv_reference,
                                            'ap_direction':virus_ap_direction,
                                            'ml_direction':virus_ml_direction,
                                            'dv_direction':virus_dv_direction}
                        coordinatedata = {'ap_location':virus_ap_location,
                                            'ml_location':virus_ml_location,
                                            'dv_location':virus_dv_location,
                        }  
                        coordinatedata.update(coordinatesystemdata)
                        try:
                            lab.CoordinateSystem().insert1(coordinatesystemdata)
                        except dj.errors.DuplicateError:
                            print('duplicate coordinate system already exists: ', coordinatesystemdata) 
                        try:
                            lab.Coordinate().insert1(coordinatedata)
                        except dj.errors.DuplicateError:
                            print('duplicate coordinate already exists: ', coordinatedata)
                        

                        virusinjdata = {
                                'surgery_id': surgeryidx,
                                'subject_id':item['animal#'],
                                'injection_id':injidx,
                                #'virus_id':virus_id,
                                'volume':virus_volume,
                                'virus_injection_remarks': 'virus injection: ' + str(item['virus inj comments ('+str(virusinjidx)+')']),
                                }
                        virusinjdata.update(coordinatedata)
                        if burrhole in [True,'T','t','True','TRUE',1]:
                            burrholeid = len(lab.Surgery.BurrHole() & surgerydata)+1
                            virusinjdata['burrhole_id'] = burrholeid
                            burrholedata = {
                                'surgery_id': surgeryidx,
                                'subject_id':item['animal#'],
                                'burrhole_id':burrholeid,
                                'burrhole_description' : '',
                                'burrhole_notes':'',
                                }
                            burrholecoordinatesystemdata = coordinatesystemdata.copy()
                            burrholecoordinatesystemdata['dv_direction'] = 'DV'
                            burrholecoordinatesystemdata['dv_reference'] = 'dura'

                            burrholecoordinatedata = coordinatedata.copy()
                            burrholecoordinatedata['dv_location'] = 0
                            burrholecoordinatedata['dv_direction'] = 'DV'
                            burrholecoordinatedata['dv_reference'] = 'dura'
                            try:
                                lab.CoordinateSystem().insert1(burrholecoordinatesystemdata)
                            except dj.errors.DuplicateError:
                                print('duplicate coordinate system already exists: ', burrholecoordinatesystemdata)
                            try:
                                lab.Coordinate().insert1(burrholecoordinatedata)
                            except dj.errors.DuplicateError:
                                print('duplicate coordinate already exists: ', burrholecoordinatedata)
                            burrholedata.update(burrholecoordinatedata)

                            virusinjdata['burrhole_id'] = burrholeid
                            try:
                                lab.Surgery.BurrHole().insert1(burrholedata)
                            except dj.errors.DuplicateError:
                                print('duplicate burrhole for animal ',item['animal#'], ' already exists: ', burrholeid)


                        else:
                            virusinjdata['craniotomy_id'] = item['virus inj craniotomy id ('+str(virusinjidx)+')']
                        try:
                            lab.Surgery.VirusInjection().insert1(virusinjdata)
                        except dj.errors.DuplicateError:
                            print('duplicate virus injection for animal ',item['animal#'], ' already exists: ', injidx)
                        # let's add the virus itself. 
                        # if a virus mixture was used:
                        if len(lab.VirusMixture() & {'virus_mixture_id':virus_id})>0:
                            virusmixturedict = (lab.VirusMixture & {'virus_mixture_id': virus_id}).fetch1()
                            virusmixtureparts = lab.VirusMixture.VirusMixturePart & {'virus_mixture_id': virus_id}
                            for part in virusmixtureparts.fetch(as_dict=True):
                                virusaliquotdict = (lab.VirusAliquot & {'virus_aliquot_id': part['virus_aliquot_id']}).fetch1()
                                virusdict = (lab.Virus & {'virus_id': virusaliquotdict['virus_id']}).fetch1()
                                viruscomponentdata = {
                                    'surgery_id': surgeryidx,
                                    'subject_id':item['animal#'],
                                    'injection_id':injidx,
                                    'virus_id': virusaliquotdict['virus_id'],
                                    'effective_titer': (virusdict['titer']/virusaliquotdict['dilution']) * part['fraction'] ,
                                }
        
                                try:
                                    lab.Surgery.VirusComponent().insert1(viruscomponentdata,ignore_extra_fields=True)
                                except dj.errors.DuplicateError:
                                    print('duplicate virus injection for animal ',item['animal#'], ' already exists: ', injidx, ' for virus ', virusaliquotdict['virus_id'])





                virusinjidx += 1    
            #%
            
            surgeryidx += 1

def ingest_devices_and_calibrations(dj):
    df_calibrations = pd.read_csv(dj.config['path.metadata']+'NDNF experimenters_Calibration.csv')
    for index, row in df_calibrations.iterrows():
        
        rig = row['Rig ID']
        device = row['Device name']
        device_dict = {} # to hold device info
        device_dictionary = {'rig': rig,
                            'device': device,
                            'device_dict': device_dict,
                            }
        try:
            lab.Device().insert1(device_dictionary)
        except dj.errors.DuplicateError:
            pass
        calibration_date = datetime.strptime(row['Calibration date'], '%Y/%m/%d').date()
        calibration_id = len((lab.Device.DeviceCalibration()&{'rig': rig,'device':device}))+1
        if device == 'LoadCell':
            if len((lab.Device.DeviceCalibration()&{'rig': rig,'device':device,'calibration_date':calibration_date})) >0:
                continue # already ingested
            calibration_dict = {}
            for ax in [0,1]:
                calibration_dict[ax] = {'g':eval(row['Axis {} g'.format(ax)]),
                                'vals':eval(row['Axis {} vals'.format(ax)]),
                                'direction':row['Axis {} direction'.format(ax)]}
            calibration_dict_out = {'rig': rig,
                                    'device': device,
                                    'calibration_date': calibration_date,
                                # 'calibration_details': '',
                                    'calibration_dict':calibration_dict,
                                }
            try:
                lab.Device.DeviceCalibration().insert1(calibration_dict_out)
            except dj.errors.DuplicateError:
                pass
        else:
            print('Unknown device type for calibration ingestion: ', device)

                    