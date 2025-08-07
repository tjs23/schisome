from glob import glob
import os
from schisome import SchisomeDataSet

run_tag = 'Aug25v1'
data_paths = glob(f'datasets/*_{run_tag}.npz')

# Run DNN inference for each dataset


#data_set.add_analysis_tracks(max_missing=0.33)
#data_set.add_prediction_tracks('class_pred_all')
#data_set.reset_2d_proj()

# Make SQLite DB for each dataset

for data_path in data_paths:
  data_set = SchisomeDataSet(data_path)
  
  sqlite_path = os.path.splitext(data_path)[0]+ '.sqlite'
  
  data_set.write_database(sqlite_path)



