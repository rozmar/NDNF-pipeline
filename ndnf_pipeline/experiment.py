import datajoint as dj
from ndnf_pipeline.utils.pipeline_tools import get_schema_name
from ndnf_pipeline import lab
schema = dj.schema(get_schema_name('experiment'),locals())

@schema
class Session(dj.Manual):
    definition = """
    -> lab.Subject
    session: smallint 		# session number - A session is defined as a recording/behavior session with a maximum of 60 minutes break between recordings
    ---
    session_date: date
    session_time: time       # session start time, from the start of the first behavior trial, if available, otherwise from the first recording file
    unique index (subject_id, session_date, session_time)
    -> lab.Person
    -> lab.Rig
    """

@schema
class Task(dj.Lookup):
    definition = """
    # Type of tasks
    task            : varchar(50)                  # task type
    ----
    task_description : varchar(4000)
    """
    contents = [
         ('DP', 'Dynamic pull task, with a single force target to reach for water rewards from a moving reward port.'),
         ]


@schema
class TaskProtocol(dj.Lookup):
    definition = """
    # SessionType
    -> Task
    task_protocol : tinyint # task protocol
    ---
    task_protocol_description : varchar(4000)
    """
    contents = [
         ('DP', 0, 'Dynamic pull basic protocol'),    
         ]


@schema
class SessionTrial(dj.Imported):
    definition = """
    -> Session
    trial : smallint 		# trial number
    ---
    trial_start_time : decimal(10, 5)  # (s) relative to session beginning 
    trial_end_time : decimal(10, 5)  # (s) relative to session beginning 
    """
  
@schema 
class TrialNoteType(dj.Lookup):
    definition = """
    trial_note_type : varchar(20)
    """
    contents = zip(('autolearn', 'protocol #', 'bad', 'autowater'))


@schema
class TrialNote(dj.Manual):
    definition = """
    -> SessionTrial
    -> TrialNoteType
    ---
    trial_note  : varchar(255) 
    """


@schema
class SessionTask(dj.Manual):
    definition = """
    -> Session
    -> TaskProtocol
    """

@schema
class SessionComment(dj.Manual):
    definition = """
    -> Session
    session_comment : varchar(1000)
    """

@schema
class SessionDetails(dj.Manual):
    definition = """
    -> Session
    session_weight : decimal(8,4) # weight of the mouse at the beginning of the session
    session_water_earned : decimal(8,4) # water earned by the mouse during the session
    session_water_extra : decimal(8,4) # extra water provided after the session
    """

@schema
class TaskSettings(dj.Imported):
    definition = """
    -> Session
    task_setting_id : smallint
    ---
    target_force_lut: longblob  # 2D array mapping force to speed (mm/s)
    target_force_axes: longblob  # 2x1D array of force values corresponding to the LUT
    reward_port_moving_speed: decimal(6,3)  # (mm/s)
    reward_port_start_pos: decimal(6,3)  # (mm)
    reward_port_end_pos: decimal(6,3)  # (mm)
    reward_size: decimal(6,3)  # (ul)
    """

@schema
class Outcome(dj.Lookup):
    definition = """
    outcome : varchar(32)
    """
    contents = zip(('hit', 'miss', 'ignore'))


@schema
class BehaviorTrial(dj.Imported):
    definition = """
    -> SessionTrial
    ----
    -> TaskProtocol
    -> TaskSettings
    -> Outcome
    """


@schema
class TrialEventType(dj.Lookup):
    definition = """
    trial_event_type  : varchar(20)  
    """
    contents = zip(('go', 'threshold crossing', 'trial end', 'reward', 'lick'))


@schema
class TrialEvent(dj.Imported):
    definition = """
    -> BehaviorTrial 
    -> TrialEventType
    trial_event_id: smallint
    ---
    trial_event_time : decimal(9, 5)   # (s) from trial start, not session start
    trial_event_duration : decimal(9,5)  #  (s)  
    """

@schema
class TrialForceTrace(dj.Imported):
    definition = """
    -> BehaviorTrial 
    force_axis_idx: tinyint  # 0: x-axis, 1: y-axis (??)
    ---
    force_trace_time: longblob  # (s) from trial start
    force_trace_value: longblob  # (g)
    """
