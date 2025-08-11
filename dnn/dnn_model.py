import os
import numpy as np
import keras

from keras import layers, ops
from tensorflow_probability.substrates import jax as tfp
from generator import MixedLocReconstructDataGenerator

     
# # # #  Custom losses and metrics  # # # #


def mix_acc_mask(y_true, y_pred, weights):
    
    is_classified = ops.cast(ops.max(y_true, axis=-1, keepdims=False) > 0.0, dtype='float32')
    
    klass_true = ops.amax(y_true, axis=-1)
    klass_pred = ops.amax(y_pred, axis=-1)
    
    is_close = ops.cast(klass_true == klass_pred, dtype='float32')
    
    return ops.divide_no_nan(ops.sum(is_classified * is_close), ops.sum(is_classified))


def mae_acc(y_true, y_pred, weights):
    
    is_close = ops.cast(ops.abs(y_true-y_pred) < 0.1, dtype='float32')
 
    return ops.mean(is_close)


def mae_mask(x_in, y_true, y_pred, weights):
    
    masked = ops.cast(x_in == 0.0, dtype='float32') * ops.cast(y_true != 0.0, dtype='float32')
    
    diff = masked * weights * ops.abs(y_true-y_pred)
    
    return ops.divide_no_nan(ops.sum(diff), ops.sum(masked * weights))


def nonzero_mae_loss(y_true, y_pred, weights):
    
    valid = ops.cast(y_true > 0.0, dtype='float32') # Not NaN
    
    diff =    weights * ops.abs(y_true-y_pred)

    return ops.divide_no_nan(ops.sum(valid * diff), ops.sum(valid * weights))
    

def nonzero_mae(x_in, y_true, y_pred, weights):

    unmasked = ops.cast(x_in > 0.0, dtype='float32') # Not mask and not NaN
    
    diff = unmasked * weights * ops.abs(y_true-y_pred)
    
    return ops.divide_no_nan(ops.sum(diff), ops.sum(unmasked * weights))


def neg_log_likelihood(y_true, y_pred, weights):
    
    distrib = tfp.distributions.Multinomial(2, logits=y_pred)
    
    nll = distrib.log_prob(y_true)
    
    nll = -ops.divide_no_nan(ops.sum(nll * weights), ops.sum(weights))
    
    return    nll / 2e2


def plot_model(image_path, data_set, batch_size=32, ndim_compress=10, nlayers_att=4, nheads_att=2):
    
    profiles = data_set.train_profiles
    klasses = data_set.train_markers   
    all_idx = range(len(profiles))
    
    data_generator = MixedLocReconstructDataGenerator(all_idx, profiles, klasses, data_set.replica_cols,
                                                      batch_size=batch_size, training=False, n_mix=0)
   
    model = get_model(data_generator, ndim_compress, nlayers_att, nheads_att)

    keras.utils.plot_model(model, to_file=image_path, show_shapes=False,
                           show_dtype=False,  show_layer_names=False,
                           rankdir='TB',  expand_nested=False,
                           dpi=200, show_layer_activations=True,
                           show_trainable=False)
                           

def make_inference(model_path, profiles, replica_cols, klasses=None, batch_size=512,
                   ndim_compress=10, nlayers_att=2, nheads_att=2, n_rep=100):
    
    all_idx = range(len(profiles))
    data_generator = MixedLocReconstructDataGenerator(all_idx, profiles, klasses, replica_cols,
                                                      batch_size=batch_size, training=False, n_mix=0)
    
    n_classes = data_generator.n_classes
    n, p = profiles.shape
    out_array_r = np.zeros((n, n_rep, p), np.float32)
    out_array_c = np.zeros((n, n_rep, n_classes), np.float32)
    latent_array = np.zeros((n, n_rep, ndim_compress))
    
    model = get_model(data_generator, n_classes, ndim_compress,
                      nlayers_att, nheads_att, add_noise=False)
    model.load_weights(model_path)
 
    ref_profiles = data_generator.ref_profiles
    ref_profiles = ops.convert_to_tensor(ref_profiles)

    n = len(all_idx)
    i = 0
    
    print(f'Making inference for {n:,} profiles')
    
    for batch, (x_in, y_true, weights) in enumerate(data_generator):
        print(f' .. {i:,}')
        j = min(n, i+batch_size)
        
        for r in range(n_rep):
            y_pred_r, latent, y_pred_c = model([x_in, ref_profiles], training=False)
                         
            cat_probs = np.exp(y_pred_c[:j-i]) # Convert from output log probs
            cat_probs /= cat_probs.sum(axis=-1)[:,None]
            
            latent_array[i:j,r] = latent[:j-i]
            out_array_c[i:j,r] = cat_probs
            out_array_r[i:j,r] = y_pred_r[:j-i]

        i = j
    
    return out_array_c, out_array_r, latent_array


def get_model(data_generator, ndim_compress=10, nlayers_att=4, nheads_att=2, noise_sigma=0.05):
    
    n_classes = data_generator.n_classes # One more than input markers/labels given on-the-fly null class

    n_ref, prof_dim =  data_generator.ref_profiles.shape[1:]
 
    x_query_in = keras.Input(shape=(1, prof_dim), name='in_prof') # Queries, to augment. Sequence lengths is (1,) : batch will be (b, 1, p)
    x_keys_in  = keras.Input(shape=(n_ref, prof_dim), name='in_ref')  # Keys, to comare to, invariant over epoch : batch will be (1, n, p)

    compress_layer = layers.Conv1D(ndim_compress, 1, name='compress', activation='gelu', padding='valid')
 
    x1 = compress_layer(x_query_in)
    x2 = compress_layer(x_keys_in)
 
    x1 = layers.GaussianNoise(noise_sigma, name=f'noise')(x1)
    x1 = layers.Dropout(0.1)(x1)
    
    x2 = ops.repeat(x2, data_generator.batch_size, axis=0) # Expand proteome ref profiles over minibatch 

    for i in range(nlayers_att):
        layer = layers.MultiHeadAttention(nheads_att, ndim_compress, dropout=0.1, name=f'MHAttention_{i+1}')
        x1 = layer(x1, x2, training=data_generator.training)
   
    #x1 = layers.LayerNormalization()(x1)
    x1 = latent = layers.Reshape((ndim_compress,), name='reshape')(x1)

    # Reconstruction

    x1_recon = tfp.layers.DenseReparameterization(ndim_compress, name='bnn_expand1', activation='gelu')(x1)
    x1_recon = layers.LayerNormalization()(x1_recon)
    
    recon_out = tfp.layers.DenseReparameterization(prof_dim, name='bnn_expand12', activation='softplus')(x1_recon)

    # Class mixtures

    x1_class = tfp.layers.DenseReparameterization(ndim_compress, name='bnn_pred1', activation='gelu')(x1)
    x1_class = layers.LayerNormalization()(x1_class)
    
    x1_class = tfp.layers.DenseReparameterization(ndim_compress, name='bnn_pred2', activation='gelu')(x1_class)
    x1_class = layers.LayerNormalization()(x1_class)
    
    class_mix_out = tfp.layers.DenseReparameterization(n_classes, name='pred_out', activation='linear')(x1_class)
 
    model = keras.Model([x_query_in, x_keys_in], [recon_out, latent, class_mix_out])
 
    return model
 
 
 
def train_model(model_path, test_idx, train_idx, profiles, replica_cols, klasses=None,
                n_mix=500, n_epochs=100, batch_size=32, init_learning_rate=1e-3,
                ndim_compress=10, nlayers_att=4, nheads_att=2):
  
  # Test and train on-the-fly data generators to create mixed profiles and classes 
  data_generator = MixedLocReconstructDataGenerator(train_idx, profiles, klasses, replica_cols, n_mix=n_mix)
  data_generator_test = MixedLocReconstructDataGenerator(test_idx, profiles, klasses, replica_cols, training=False, n_mix=n_mix)

  model = get_model(data_generator, ndim_compress, nlayers_att, nheads_att)

  decay_steps = n_epochs * data_generator.n_batches  
  learning_rate = keras.optimizers.schedules.CosineDecay(init_learning_rate, decay_steps, alpha=0.01)  
  optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

  metrics = {'acc_cat':mix_acc_mask, 'acc_recon':mae_acc,
             'mae_recon':nonzero_mae, 'mae_mask':mae_mask}
  
  losses = {'loss_class':neg_log_likelihood ,'loss_recon':nonzero_mae_loss}
  
  model.compile(loss=losses, optimizer=optimizer, metrics=metrics)

  history = model.fit(data_generator, epochs=n_epochs, batch_size=batch_size, validation_data=data_generator_test)

  model.save_weights(model_path)
  
  return history

   
