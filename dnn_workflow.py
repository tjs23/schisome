import os
#os.environ["KERAS_BACKEND"] = "jax"
import numpy as np
from glob import glob
from schisome import SchisomeDataSet
from dnn import dnn_model
from general_util import info, plot_training_history

run_tag = 'Aug25v1'
data_paths = glob(f'datasets/*_{run_tag}.npz')

overwrite = False
n_models = 10

nlayers_att = 4
batch_size = 32
n_epochs = 75
n_infer_samples = 100  #  per model 

for data_path in data_paths:
  
    data_set = SchisomeDataSet(data_path)
    
    if data_set.has_predictions and not overwrite:
       info(f'Prediction data already present for {data_path}')
       continue
       
    prof_dim = data_set.train_profiles.shape[-1]
 
    if prof_dim > 64:
        ndim_compress = 64
    elif prof_dim > 32:
        ndim_compress = 32
    else:
        ndim_compress = 16
      
    max_nan = int(0.35 * prof_dim)    
    
    run_tag = os.path.splitext(os.path.basename(data_path))[0]
    
    model_paths = [f'models/DNN_Model_v{m:02d}_{run_tag}.weights.h5' for m in range(n_models)]
    
    info(f'Fetching ensemble (size {n_models}) test/train data for {data_path}')
    
    valid, idx_chunks, valid_profiles, valid_class_labels = data_set.get_train_test_data(n_chunks=n_models, max_nan=max_nan)
    
    n = len(valid)
    m = np.count_nonzero(valid)
    
    marker_klasses = []
    marker_idx = np.zeros(n)
    valid_marker_idx = np.zeros(m)
    
    for m in range(n_models):
        test_idx = idx_chunks[m]
        valid_marker_idx[test_idx] = m+1
        marker_klasses.append(f'group_{m+1}')
    
    marker_idx[valid] = valid_marker_idx
    
    # Remember test-train split ; store excluded chunk index
    
    data_set.set_marker_data(data_set.train_groups_key, marker_idx, marker_klasses)
    
    # Train models
        
    for m, model_path in enumerate(model_paths):
        model_path = model_paths[m]
        
        if os.path.exists(model_path) and not overwrite:
            info(f'Found existing {model_path}')
            continue
            
        test_idx = idx_chunks[m]
        train_idx = np.concatenate(idx_chunks[:m] + idx_chunks[m+1:])    
        
        info(f'Training model {m+1} with {len(train_idx):,} profiles, testing with {len(test_idx):,}')
        
        history = dnn_model.train_model(model_path, test_idx, train_idx, valid_profiles, valid_class_labels,
                                        data_set.replica_cols, n_mix=250, n_epochs=n_epochs,
                                        batch_size=batch_size, ndim_compress=ndim_compress, nlayers_att=nlayers_att)
    
        info(f'Saved {model_path}')

        history_path = os.path.splitext(model_path)[0] + '_training.png'
 
        plot_training_history(history, file_path=history_path)

    # Make inference for whole proteome for each model

    profiles = data_set.train_profiles # Inference on everything, no nan filter
    klasses = data_set.train_markers
    
    pred_classes = []
    recon_profiles = []
    latent_profiles = []
    
    for m, model_path in enumerate(model_paths):
        info(f'Working on model {m} : {model_path}')
        class_vecs, recon_vecs, latent_vecs = dnn_model.make_inference(model_path, profiles, data_set.replica_cols,
                                                                       klasses, ndim_compress=ndim_compress,
                                                                       nlayers_att=nlayers_att, n_rep=n_infer_samples)
 
        pred_classes.append(class_vecs)
        recon_profiles.append(recon_vecs)
        latent_profiles.append(latent_vecs)
    
    # Aggregate and stor model outputs
    info(f'Aggregating results to {data_path}')
    
    # combine along replica axis
    pred_classes = np.concatenate(pred_classes, axis=1)
    recon_profiles = np.concatenate(recon_profiles, axis=1)
    latent_profiles = np.concatenate(latent_profiles, axis=1)
 
    # Average profiles over training ensemble
    recon_profiles  = np.median(recon_profiles, axis=1)
    latent_profiles = latent_profiles.mean(axis=1)
    
    data_set.set_profile_data(data_set.recon_profile_key, recon_profiles)
    data_set.set_profile_data(data_set.latent_profile_key, latent_profiles)
 
    # Ensemble class predictions
    data_set.set_pred_class_data(data_set.class_ensemble_key, pred_classes, data_set.train_markers)
    
    


