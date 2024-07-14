import os

USER_HOME = os.environ.get('HOME')
DEFAULT_ENVS = dict(
    DLX_HOME=os.path.join(USER_HOME, '.cache/dlx')
)

DLX_HOME=os.environ.get('DLX_HOME', DEFAULT_ENVS['DLX_HOME'])
if not os.path.exists(DLX_HOME):
    os.mkdir(DLX_HOME)