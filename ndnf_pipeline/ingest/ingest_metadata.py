#%% experimenters
import pandas as pd
#import ndnf_pipeline.lab as lab
from .. import lab
def ingest_metadata(dj):
    ingest_experimenters(dj)
    ingest_rigs(dj)
    ingest_viruses(dj)
    ingest_mouse_lines(dj)

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