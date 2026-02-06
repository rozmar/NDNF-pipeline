
import datajoint as dj
from ndnf_pipeline.utils.pipeline_tools import get_schema_name
import subprocess
import os
import getpass
import socket
from datetime import datetime

schema = dj.schema(get_schema_name('analysis_log'),locals())

@schema
class ExecutionLog(dj.Manual):
    definition = """
    execution_log_id : int
    ---
    execution_timestamp : datetime
    script_name         : varchar(255)
    git_hash            : varchar(40)
    git_status          : longblob      # output of git status --porcelain
    git_diff            : longblob      # output of git diff
    user_name           : varchar(64)
    host_name           : varchar(64)
    working_directory   : varchar(512)
    python_version      : varchar(64)
    environment_name    : varchar(255)
    installed_packages  : longblob      # output of pip freeze
    """

def log_execution(script_name=''):
    """
    Captures the current execution context and inserts it into ExecutionLog.
    Returns the execution_log_id.
    """
    import sys
    import re
    
    timestamp = datetime.now()
    user = getpass.getuser()
    host = socket.gethostname()
    cwd = os.getcwd()
    
    # Environment info
    python_version = sys.version.split(' ')[0]
    
    # Check for UV/Venv first, then Conda
    env_name = os.environ.get('VIRTUAL_ENV')
    if env_name:
        env_name = os.path.basename(env_name) # Just the folder name
    else:
        env_name = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    
    try:
        # Try uv pip freeze first (primary method)
        installed_packages = subprocess.check_output(['uv', 'pip', 'freeze', '--color=never'], stderr=subprocess.STDOUT).strip().decode('utf-8')
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # Fallback 1: python -m pip freeze
            installed_packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], stderr=subprocess.STDOUT).strip().decode('utf-8')
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                # Fallback 2: direct pip freeze
                installed_packages = subprocess.check_output(['pip', 'freeze'], stderr=subprocess.STDOUT).strip().decode('utf-8')
            except Exception as e:
                print(f"Warning: Could not capture installed packages: {e}")
                installed_packages = f"Failed to capture packages: {e}"
    
    # Strip ANSI color codes if present (just in case --color=never isn't respected or finding its way in)
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    installed_packages = ansi_escape.sub('', installed_packages)

    try:
        # Get git info
        # Assuming the pipeline is in a git repo
        repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
        if not os.path.exists(os.path.join(repo_path, '.git')):
             # Fallback if not running from expected structure, try cwd
             repo_path = cwd
        
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo_path).strip().decode('utf-8')
        git_status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo_path).strip().decode('utf-8')
        git_diff = subprocess.check_output(['git', 'diff'], cwd=repo_path).strip().decode('utf-8')
        
    except Exception as e:
        print(f"Warning: Could not capture git info: {e}")
        git_hash = 'unknown'
        git_status = str(e)
        git_diff = ''

    # Manually assign execution_log_id
    execution_log_id = len(ExecutionLog()) + 1

    key = {
        'execution_log_id': execution_log_id,
        'execution_timestamp': timestamp,
        'script_name': script_name,
        'git_hash': git_hash,
        'git_status': git_status.encode('utf-8') if git_status else b'',
        'git_diff': git_diff.encode('utf-8') if git_diff else b'', # using blob for potentially large diff
        'user_name': user,
        'host_name': host,
        'working_directory': cwd,
        'python_version': python_version,
        'environment_name': env_name,
        'installed_packages': installed_packages.encode('utf-8') if installed_packages else b''
    }
    
    ExecutionLog.insert1(key)
    
    return execution_log_id

