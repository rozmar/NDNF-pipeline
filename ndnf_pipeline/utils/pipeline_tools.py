
#import logging

import datajoint as dj
#import hashlib
#log = logging.getLogger(__name__)


def get_schema_name(name):
    if name in ['lab','experiment','analysis_log','behavior_analysis','videography','environment']:
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
    
    schema = dj.schema(schemaname+'_experiment')
    schema.drop(force=True) 
    schema = dj.schema(schemaname+'_lab')
    schema.drop(force=True) 
    schema = dj.schema(schemaname+'_analysis_log')
    schema.drop(force=True) 

def get_github_commit_url(git_remote_url, git_hash, file_path=None):
    """
    Convert git remote URL and hash to a GitHub commit or file URL.
    
    Supports both SSH and HTTPS formats:
    - git@github.com:user/repo.git
    - https://github.com/user/repo.git
    
    Parameters:
    -----------
    git_remote_url : str
        Git remote origin URL
    git_hash : str
        Git commit hash
    file_path : str, optional
        Relative path to file from repo root. If provided, links to the specific file.
    """
    import re
    
    if not git_remote_url or git_hash == 'unknown':
        return None
    
    # SSH format: git@github.com:user/repo.git
    ssh_match = re.match(r'git@github\.com:(.+)/(.+)\.git', git_remote_url)
    if ssh_match:
        owner, repo = ssh_match.groups()
        base_url = f"https://github.com/{owner}/{repo}"
    else:
        # HTTPS format: https://github.com/user/repo.git
        https_match = re.match(r'https://github\.com/(.+)/(.+?)(?:\.git)?$', git_remote_url)
        if https_match:
            owner, repo = https_match.groups()
            base_url = f"https://github.com/{owner}/{repo}"
        else:
            return None
    
    # If file_path provided, link to the specific file
    if file_path:
        return f"{base_url}/blob/{git_hash}/{file_path}"
    else:
        return f"{base_url}/commit/{git_hash}"


def view_session_code(session_key):
    """
    View the code version used for a specific session.
    
    Parameters:
    -----------
    session_key : dict
        Dictionary with 'subject_id' and 'session', e.g., {'subject_id': 'NDNF300', 'session': 1}
    """
    from ndnf_pipeline import experiment, analysis_log
    import os
    
    # Get execution log
    exec_log_id = (experiment.SessionProvenance & session_key).fetch1('execution_log_id')
    log = (analysis_log.ExecutionLog & {'execution_log_id': exec_log_id}).fetch1()
    
    print(f"=== Session Code Version ===")
    print(f"Session: {session_key}")
    print(f"Timestamp: {log['execution_timestamp']}")
    print(f"User: {log['user_name']}@{log['host_name']}")
    print(f"Script: {log['script_name']}")
    print(f"\nGit Hash: {log['git_hash']}")
    print(f"Python: {log['python_version']}")
    print(f"Environment: {log['environment_name']}")
    
    # Generate GitHub link to the specific file
    script_path = log['script_name']
    working_dir = log['working_directory']
    
    # Compute relative path from repo root
    # Assuming the working directory is the repo root or close to it
    relative_path = None
    if script_path and working_dir:
        try:
            # Try to get relative path
            if script_path.startswith(working_dir):
                relative_path = os.path.relpath(script_path, working_dir)
            else:
                # Try to find common repo structure (e.g., ndnf_pipeline/)
                if 'ndnf_pipeline' in script_path:
                    idx = script_path.find('ndnf_pipeline')
                    relative_path = script_path[idx:]
        except:
            pass
    
    github_url = get_github_commit_url(log['git_remote_url'], log['git_hash'], relative_path)
    if github_url:
        print(f"\n🔗 GitHub File: {github_url}")
    
    print(f"\n=== Git Status ===")
    git_status = log['git_status'].decode('utf-8') if isinstance(log['git_status'], bytes) else log['git_status']
    print(git_status or "Clean working directory")
    
    diff = log['git_diff'].decode('utf-8') if isinstance(log['git_diff'], bytes) else log['git_diff']
    if diff:
        print(f"\n=== Uncommitted Changes ===")
        print(diff)
    else:
        print(f"\n=== No uncommitted changes ===")
    
    return log