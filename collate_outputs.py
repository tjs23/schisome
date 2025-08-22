from glob import glob
import os
from schisome import SchisomeDataSet

run_tag = 'Aug25v1'
data_paths = glob(f'datasets/*_{run_tag}.npz')

## ## Add Miguel's experimental lists

for data_path in data_paths:
   data_set = SchisomeDataSet(data_path)
   data_set.info(f'Finalising {data_path}')  
   data_set.reset_2d_proj()
   plot_args = dict(min_nonzero=0.9, spot_size=16)
   
   if not data_set.has_predictions:
       data_set.warn(f'Dataset at {data_path} has no mixed class and reconstruction predictions. The DNN workflow should be run first.')
       continue
   
   if not data_set.has_array_key('pval'):
       data_set.info(f'Calculating p-values for {data_path}')
       data_set.make_pvalues()

   if not data_set.has_array_key('prediction'):
       data_set.info(f'Making class predictions for {data_path}')
       data_set.make_class_predictions()
           
   if not data_set.has_profile_key('zfill'):
       data_set.info(f'Reconstructing profiles for {data_path}')
       data_set.make_profile_predictions(max_missing=0.35)

   data_set.plot_umap_2d('zfill', ['training', 'prediction', 'prediction2'], ['Train', 'Pred', 'Duals'], **plot_args)
   data_set.plot_umap_2d('recon', ['training', 'prediction', 'prediction2'], ['Train', 'Pred', 'Duals'], **plot_args)

   file_root = os.path.splitext(data_path)[0]
   sqlite_path = file_root + '.sqlite' 
   data_set.info(f'Making SQLite3 database {sqlite_path}')  
   data_set.write_database(sqlite_path)

   profile_tsv_path = file_root + '_profiles.tsv'    
   data_set.write_profile_tsv(profile_tsv_path, profiles=['init','zfill','latent'],
                              markers=['training','prediction','prediction2'])
