import datajoint as dj
from ndnf_pipeline.utils.pipeline_tools import get_schema_name
from ndnf_pipeline import lab

schema = dj.schema(get_schema_name('environment'),locals())


@schema
class EnvSensor(dj.Manual):
    definition = """
    sensor_id               : varchar(32)   # e.g. NDNF-#1
    ---
    sensor_description = '' : varchar(256)
    """
    class EnvSensorPlacement(dj.Part):
        # mirrors lab.Rig.RigState - one row per time the sensor was (re)placed
        definition = """
        -> master
        placement_date : date
        ---
        -> lab.Institute
        building               : varchar(50)
        room                   : varchar(20)
        -> [nullable] lab.Rig                   # only when tied to a specific rig
        location_description   : varchar(255)   # "Marton's desk", "top of ventilated box"
        placement_comment = '' : varchar(512)
        """

@schema
class EnvSensorChannel(dj.Lookup):
    definition = """
    channel_name : varchar(32)   # Temperature, Humidity, Light, Voltage, WIFI_RSSI
    ---
    unit = ''    : varchar(16)
    """
    contents = [
        ('Temperature', 'C'),
        ('Humidity',    '%'),
        ('Light',       'lux'),
        ('Voltage',     'V'),
        ('WIFI_RSSI',   'dBm'),
    ]

@schema
class EnvSensorRecording(dj.Manual):
    definition = """
    -> EnvSensor
    recording_datetime : datetime   # hourly bin start
    """
    class Channel(dj.Part):
        definition = """
        -> master
        -> EnvSensorChannel
        ---
        value_avg    : float
        value_min    : float
        value_max    : float
        sample_count : smallint
        """