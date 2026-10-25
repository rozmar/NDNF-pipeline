import datajoint as dj
from ndnf_pipeline.utils.pipeline_tools import get_schema_name
from ndnf_pipeline import lab
from ndnf_pipeline import analysis_log

schema = dj.schema(get_schema_name('experiment'),locals())

@schema
class Session(dj.Manual):
    definition = """
    -> lab.Subject
    session: smallint 		# session number - A session is defined as a recording/behavior session with a maximum of 60 minutes break between recordings
    ---
    session_date: date
    session_time: time       # session start time, from the start of the first behavior trial, if available, otherwise from the first recording file
    -> lab.Person
    -> lab.Rig
    """

@schema
class SessionProvenance(dj.Manual):
    definition = """
    -> Session
    ---
    -> analysis_log.ExecutionLog
    """

@schema
class SessionTrial(dj.Manual):
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
class ForceDirection(dj.Lookup):
    definition = """
    force_direction  : varchar(2)
    ---
    force_direction_description : varchar(255)
    """
    contents = [('AP', 'anterior-posterior'), 
                ('PA', 'posterior-anterior'),
                ('LR', 'left-right'),
                ('RL', 'right-left')]

@schema
class TaskSettings(dj.Manual):
    definition = """
    -> Session
    task_setting_id : smallint
    ---
    target_force_lut: longblob  # 2D array mapping force to speed (mm/s)
    reward_port_start_pos: decimal(6,3)  # (mm)
    reward_port_end_pos: decimal(6,3)  # (mm)
    reward_size: decimal(6,3)  # (ul)
    """
    class ForceAxis(dj.Part):
        definition = """
        -> master
        force_axis_idx: tinyint  # 0: x-axis, 1: y-axis
        ---
        -> ForceDirection
        target_force_axes: longblob  # 1D array of force values corresponding to the force_axis_idx of the LUT

        """

@schema
class FeedbackType(dj.Lookup):
    definition = """
    feedback_type: varchar(32)
    ---
    feedback_dimensions: tinyint   # 1, 2, 0=boolean/unspecified
    feedback_modality: varchar(20) # 'speed', 'position', 'unspecified'
    feedback_description: varchar(255)
    """
    contents = [
        ('1D_speed',     1, 'speed',       '1D accumulating/speed feedback'),
        ('1D_position',  1, 'position',    '1D instantaneous/position feedback'),
        ('2D_speed',     2, 'speed',       '2D accumulating/speed feedback'),
        ('2D_position',  2, 'position',    '2D instantaneous/position feedback'),
        ('0D_position',  0, 'position',     'Reward-only boolean feedback with position'),
        ('0D_speed',     0, 'speed',       'Reward-only boolean feedback with speed'),
    ]

@schema
class Block(dj.Manual):
    definition = """
    -> Session
    block: smallint  # block number within session, ordered chronologically
    ---
    -> TaskSettings
    -> FeedbackType
    block_start_time: decimal(10, 5)  # (s) relative to session beginning
    block_end_time: decimal(10, 5)  # (s) relative to session beginning
    """

@schema
class SessionStructure(dj.Computed):
    definition = """
    -> Session
    ---
    n_blocks: tinyint        # total number of blocks
    n_trials: smallint       # total number of trials
    n_task_settings: tinyint # distinct task settings across blocks
    n_feedback_types: tinyint # distinct (non-unspecified) feedback types across blocks
    n_conditions: tinyint    # distinct (task_setting, feedback_type) combinations
    """

    def make(self, key):
        task_setting_ids, feedback_types = (Block & key).fetch('task_setting_id', 'feedback_type')
        n_trials = len(SessionTrial & key)
        specified = [f for f in feedback_types if f != 'unspecified']
        self.insert1({**key,
                      'n_blocks': len(task_setting_ids),
                      'n_trials': n_trials,
                      'n_task_settings': len(set(task_setting_ids)),
                      'n_feedback_types': len(set(specified)) if specified else 0,
                      'n_conditions': len(set(zip(task_setting_ids, feedback_types)))})

@schema
class Outcome(dj.Lookup):
    definition = """
    outcome : varchar(32)
    """
    contents = zip(('hit', 'miss', 'ignore'))


@schema
class BehaviorTrial(dj.Manual):
    definition = """
    -> SessionTrial
    ----
    -> Block
    -> Outcome
    """


@schema
class TrialEventType(dj.Lookup):
    definition = """
    trial_event_type  : varchar(20)  
    """
    contents = zip(('go', 'threshold crossing', 'trial end', 'reward', 'lick'))


@schema
class TrialEvent(dj.Manual):
    definition = """
    -> BehaviorTrial 
    -> TrialEventType
    trial_event_id: smallint
    ---
    trial_event_time : decimal(9, 5)   # (s) from trial start, not session start
    trial_event_duration : decimal(9,5)  #  (s)  
    """

@schema
class TrialMetrics(dj.Computed):
    definition = """
    -> BehaviorTrial
    ---
    trial_length: float          # (s) trial_end_time - trial_start_time
    time_to_reward = null: float # (s) from trial start to threshold crossing, null if unrewarded
    """

    def make(self, key):
        trial_start, trial_end = (SessionTrial & key).fetch1('trial_start_time', 'trial_end_time')
        trial_length = float(trial_end) - float(trial_start)

        events = (TrialEvent & key & {'trial_event_type': 'threshold crossing'}).fetch(
            'trial_event_time', order_by='trial_event_id'
        )
        time_to_reward = float(events[0]) if len(events) > 0 else None

        self.insert1({**key, 'trial_length': trial_length, 'time_to_reward': time_to_reward})


@schema
class TrialForceTrace(dj.Manual):
    definition = """
    -> BehaviorTrial
    ---
    force_trace_time: longblob  # (s) from trial start
    """
    class TrialForceAxis(dj.Part):
        definition = """
        -> master
        force_axis_idx: tinyint  # 0: x-axis, 1: y-axis (??)
        ---
        force_trace_value: longblob  # (g)
        """
@schema
class TrialRewardPortPosition(dj.Manual):
    definition = """
    -> BehaviorTrial 
    ---
    reward_port_position_time: longblob  # (s) from trial start
    reward_port_position_values: longblob  # (mm)
    """