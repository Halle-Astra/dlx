import os
import sys

homepath_name = 'HOME' if sys.platform == 'linux' else 'HOMEPATH'
USER_HOME = os.environ.get(homepath_name, '.')
DEFAULT_ENVS = dict(
    DLX_HOME=os.path.join(USER_HOME, '.cache/dlx')
)

DLX_HOME=os.environ.get('DLX_HOME', DEFAULT_ENVS['DLX_HOME'])
if not os.path.exists(DLX_HOME):
    os.mkdir(DLX_HOME)