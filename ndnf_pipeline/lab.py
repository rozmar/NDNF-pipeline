import datajoint as dj
from ndnf_pipeline.utils.pipeline_tools import get_schema_name

schema = dj.schema(get_schema_name('lab'),locals())

@schema
class Person(dj.Manual):
    definition = """
    user_name : varchar(24) 
    ----
    full_name : varchar(255)
    """


@schema
class Rig(dj.Manual): # TODO this doesn't capture details of the rig
    definition = """
    rig             : varchar(24)
    """
    class RigState(dj.Part):
        definition = """
        # State of various rig components
        -> master
        rig_update_date : date
        ---
        building        : varchar(50) # example KOKI tower
        room            : varchar(20) # example 2w.342
        rig_description : varchar(1024) 
        """

@schema
class MouseLine(dj.Manual):
    definition = """
    mouse_line : varchar(60)  # e.g., 'Pvalb-IRES-Cre', 'Ai14', 'NDNF-Cre'
    ---
    line_description : varchar(1024)
    jax_stock_number = null : int
    """
    
@schema
class Subject(dj.Manual):
    definition = """
    subject_id : varchar(64)
    ---
    -> [nullable] Person
    cage_number =null : int
    date_of_birth     : date
    sex               : enum('M','F','Unknown')
    """
    
    class Lineage(dj.Part):
        """Which mouse lines this animal came from with zygosity"""
        definition = """
        -> master
        -> MouseLine
        ---
        zygosity : enum('Het', 'Hom', 'Negative', 'Unknown')
        """


@schema
class WaterRestriction(dj.Manual):
    definition = """
    -> Subject
    ---
    water_restriction_id    : varchar(16)   # WR number
    wr_start_date               : date
    wr_start_weight             : Decimal(6,3)
    """
    class WaterRestrictionLog(dj.Part):
        definition = """
        # Daily log of water restriction
        -> master
        log_date        : date
        ---
        weight          : Decimal(6,3)   # weight on that day, grams
        water_intake    : Decimal(6,3)   # water intake on that day (ml)
        notes           : varchar(256)
        """


@schema
class VirusSource(dj.Lookup):
    definition = """
    virus_source   : varchar(60)
    """
    contents = [['Janelia'], ['UPenn'], ['Addgene'], ['UNC'], ['Other']]


@schema
class Serotype(dj.Manual):
    definition = """
    serotype   : varchar(60)
    """

@schema
class Virus(dj.Manual):
    definition = """
    virus_id : varchar(256)
    ---
    -> VirusSource 
    -> Serotype
    -> Person
    virus_name      : varchar(256)
    titer           : Decimal(20,1) # GC/ml
    order_date      : date
    addgene_id=null : int
    lot_number      : varchar(256)    
    virus_remarks         : varchar(256)
    """

@schema
class Solution(dj.Manual):
    definition = """
    solution_id : varchar(256)
    ---
    -> Person
    prep_date       : date
    solution_remarks         : varchar(256)
    """
    class SolutionComponent(dj.Part):
        definition = """
        # Components of virus solution
        -> master
        ingredient : varchar(256)
        ---
        amount     : varchar(256)
        """


@schema
class VirusAliquot(dj.Manual):
    definition = """
    -> Virus
    virus_aliquot_id : varchar(256)
    ---
    -> [nullable] Solution # nullable in case there is no dilution
    dilution        : Decimal (10, 2) # 1 to how much
    prep_date       : date
    virus_aliquot_remarks         : varchar(256)
    """

@schema
class VirusMixture(dj.Manual):
    definition = """
    virus_mixture_id : varchar(256)
    ---
    prep_date       : date
    virus_mixture_remarks         : varchar(256)
    """
    class VirusMixturePart(dj.Part):
        definition = """
        -> master
        -> VirusAliquot
        ---
        fraction    : Decimal(5,4)  # fraction of total volume
        """


@schema
class APMLReference(dj.Lookup):
    definition = """
    ap_ml_reference   : varchar(60)
    """
    contents = [['Bregma'], ['Lambda'], ['Allen CCF']]

@schema
class APDirection(dj.Lookup):
    definition = """
    ap_direction   : varchar(2)
    ---
    ap_direction_description : varchar(255)
    """
    contents = [('AP', 'antero-posterior'), 
                ('PA', 'postero-anterior')]
@schema
class MLDirection(dj.Lookup):
    definition = """
    ml_direction   : varchar(2)
    ---
    ml_direction_description : varchar(255)
    """
    contents = [('ML', 'medio-lateral'), 
                ('LM', 'latero-medial'), 
                ('LR', 'left-right'), 
                ('RL', 'right-left')]
                   
@schema
class DVReference(dj.Lookup):
    definition = """
    dv_reference   : varchar(60)
    """
    contents = [['dura'], ['Allen CCF']]
@schema
class DVDirection(dj.Lookup):
    definition = """
    dv_direction   : varchar(2)
    ---
    dv_direction_description : varchar(255)
    """
    contents = [('DV', 'dorso-ventral'), 
                ('VD', 'ventro-dorsal')]
@schema
class CoordinateSystem(dj.Manual):
    definition = """
    -> APMLReference
    -> DVReference
    -> APDirection
    -> MLDirection
    -> DVDirection
    """
@schema
class Coordinate(dj.Manual):    
    definition = """
    -> CoordinateSystem
    ap_location     : Decimal(8,3) # um
    ml_location     : Decimal(8,3) # um
    dv_location     : Decimal(8,3) # um 
    """
@schema
class Surgery(dj.Manual):
    definition = """
    -> Subject
    surgery_id          : int      # surgery number
    ---
    -> Person
    date                : date  # date of surgery
    start_time          : datetime # start time
    end_time            : datetime # end time
    surgery_description : varchar(256)
    surgery_notes  = '' : varchar(512)
    """

    
    class Craniotomy(dj.Part):
        definition = """
        # Other things you did to the animal
        -> master
        craniotomy_id : int
        ---
        -> Coordinate
        craniotomy_diameter : Decimal(6,3) # in mm
        craniotomy_notes  = '' : varchar(512)
        """

    class BurrHole(dj.Part):
        definition = """
        # Burr holes made during surgery
        -> master
        burrhole_id : int
        ---
        -> Coordinate
        burrhole_description  = ''   : varchar(1000) # shape and tools
        burrhole_notes  = '' : varchar(512)
        """
    class VirusInjection(dj.Part):
        definition = """
        # Virus injections
        -> master
        injection_id : int
        ---
        -> [nullable] Surgery.Craniotomy
        -> [nullable] Surgery.BurrHole
        -> Coordinate
        volume          : Decimal(10,3) # in nl
        virus_injection_remarks         : varchar(256)
        """
    class VirusComponent(dj.Part): #TODO is this final?
        definition = """
        # Components of virus injection - e.g., multiple viruses mixed
        -> master.VirusInjection
        -> Virus
        ---
        effective_titer    : Decimal(20,1) # GC/ml
        """
@schema
class Device(dj.Lookup):
    """Devices within the lab.

    Attributes:
        device ( varchar(32) ): Device short name.
        modality ( varchar(64) ): Modality for which this device is used.
        description ( varchar(256) ): Optional. Description of the device.
    """

    definition = """
    -> Rig
    device             : varchar(32)
    ---
    modality           : varchar(64)
    description=''     : varchar(256)
    """

    class DeviceCalibration(dj.Part):
        """Calibration information for devices.

        Attributes:
            device_calibration_id ( int ): Unique identifier for the calibration entry.
            calibration_date ( date ): Date when the calibration was performed.
            calibration_details ( varchar(512) ): Details about the calibration procedure.
        """

        definition = """
        -> master
        device_calibration_id : int
        ---
        calibration_date      : date
        calibration_details   : varchar(512)
        calibration_dict     : longblob
        """