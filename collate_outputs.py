from glob import glob
import os
from schisome import SchisomeDataSet

run_tag = 'Aug25v1'
data_paths = glob(f'datasets/*_{run_tag}.npz')

## ## Add Miguel's experimental lists

for data_path in data_paths:
   data_set = SchisomeDataSet(data_path)
   data_set.info(f'Finalising {data_path}')  
   
   if not data_set.has_predictions:
       data_set.warn(f'Dataset at {data_path} has no mixed class and reconstruction predictions. The DNN workflow should be run first.')
       continue
   
   if not data_set.has_marker_key('predictions'):
       data_set.info(f'Calculating p-values and reconstructing profiles for {data_path}')
       data_set.make_profile_predictions(max_missing=0.35)
       data_set.make_class_predictions()
       data_set.reset_2d_proj()

   sqlite_path = os.path.splitext(data_path)[0]+ '.sqlite' 
   data_set.info(f'Making SQLite3 database {sqlite_path}')  
   data_set.write_database(sqlite_path)



