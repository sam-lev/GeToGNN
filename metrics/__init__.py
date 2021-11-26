from .prediction_score import recall
from .prediction_score import precision
from .prediction_score import f1
from .utils import write_model_scores

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
    from .model_metrics import compute_getognn_metrics