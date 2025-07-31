#
# a central location where submodules can find the data directory.
#

import os.path

_code_dir = os.path.dirname(os.path.realpath(__file__))
_proj_dir = os.path.dirname(_code_dir)
_data_dir = os.path.join(_proj_dir, 'data')

# for lack of a better idea, just hard code the shared path
_shared_data_dir = '/data/rttools/TIRS_ancillary'
