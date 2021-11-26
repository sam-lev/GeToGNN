from __future__ import print_function
from __future__ import division
from .inits import *
from .metrics import *
from .features import *
from .LinearRegression import LinearRegression

import os
from localsetup import LocalSetup
LocalSetup = LocalSetup()
# clear past runs
model_type = os.path.join(LocalSetup.project_base_path, 'model_type.txt')
logged_model = open(model_type,'r')
m = logged_model.readlines()
logged_model.close()

print("    * read model",m)
if m[0] != 'unet':
    from .aggregators import *
    from .layers import *
    from .models import *
