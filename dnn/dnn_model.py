import os, sys, time
import numpy as np
from collections import defaultdict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras
import tensorflow as tf

from keras import layers, ops, metrics
#from tensorflow_probability.substrates import jax as tfp
import tensorflow_probability as tfp

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from .generator import MixedLocReconstructDataGenerator
from general_util import info, time_str
from constants import AnsiColors

ac = AnsiColors()

keras.utils.set_random_seed(int(time.time()))

REPORT_INTERVAL = 0.5
RECON_ACC_LIM = 0.05
EPS = 1e-9

# # # #  Custom losses and metrics  # # # #


def mix_acc_mask(y_true, y_pred, weights):
    
    is_classified = ops.cast(ops.max(y_true, axis=-1, keepdims=False) > 0.0, dtype='float32')
    
    klass_true = ops.argmax(y_true, axis=-1)
    klass_pred = ops.argmax(y_pred, axis=-1)
    
    is_close = ops.cast(klass_true == klass_pred, dtype='float32')
    
    return ops.divide_no_nan(ops.sum(is_classified * is_close), ops.sum(is_classified))


def mae_acc(y_true, y_pred, weights):
    
    is_close = ops.cast(ops.abs(y_true-y_pred) < RECON_ACC_LIM, dtype='float32')
 
    return ops.mean(is_close)


def mae_mask(x_in, y_true, y_pred, weights):
    
    masked = ops.cast(x_in == 0.0, dtype='float32') * ops.cast(y_true != 0.0, dtype='float32')
    
    diff = masked * weights * ops.abs(y_true-y_pred)
    
    return ops.divide_no_nan(ops.sum(diff), ops.sum(masked * weights))


def nonzero_mae_loss(y_true, y_pred, weights):
    
    valid = ops.cast(y_true > 0.0, dtype='float32') # Not NaN
    
    diff = weights * ops.abs(y_true-y_pred)

    return ops.divide_no_nan(ops.sum(valid * diff), ops.sum(valid * weights))
    

def nonzero_mae(x_in, y_true, y_pred, weights):

    unmasked = ops.cast(x_in > 0.0, dtype='float32') # Not mask and not NaN
    
    diff = unmasked * weights * ops.abs(y_true-y_pred)
    
    return ops.divide_no_nan(ops.sum(diff), ops.sum(unmasked * weights))


def neg_log_likelihood(y_true, y_pred, weights):
    
    true_probs = y_true + EPS # Prevent overflow from zeros
    pred_probs = y_pred + EPS
    
    nll = true_probs * ops.log(pred_probs)
    nll = -ops.divide_no_nan(ops.sum(nll * weights), ops.sum(weights))    

    return nll


def _init_metrics(loss_names, acc_names, loss_met=metrics.Mean, acc_met=metrics.Mean):

    loss_str = ' '.join([ac.yellow + name + ':' + ac.end + '{:5.3f}' for name in loss_names])
    loss_str2 = ' '.join([ac.yellow + (' ' * (len(name) + 1)) + ac.end + ac.lt_cyan + '{:5.3f}' + ac.end for name in loss_names])
    acc_str = ' '.join([ac.cyan + name + ':' + ac.end + '{:5.3f}' for name in acc_names])
    acc_str2 = ' '.join([ac.cyan + (' ' * (len(name) + 1)) + ac.end + ac.lt_cyan + '{:5.3f}' + ac.end for name in acc_names])
 
    report_line1 = ac.red + 'EP:' + ac.end + '{:3d}/{:3d} ' + ac.red + 'B:' + ac.end + '{:3d}/{:3d} ' + ac.lt_blue + 'T:' + ac.end + '{}/{} ' + loss_str + ' ' + acc_str
    report_line2 = ac.lt_blue + '                     dT:' + ac.end + '{:5.1f}ms' + ac.lt_cyan + ' VAL: ' + ac.end + loss_str2 + ' ' + acc_str2
    
    loss_metrics = [(loss_met(name=mn), loss_met(name='val_'+mn)) for mn in loss_names]     
    acc_metrics  = [(acc_met(name=f'am_{mn}'), acc_met(name=f'val_am_{mn}')) for mn in acc_names]
    
    return loss_metrics, acc_metrics, report_line1, report_line2
    
    
def _report(epoch, n_epochs, batch, n_batches, v_time, t_taken, disp_time, loss_metrics, acc_metrics, report_line1, report_line2, mean_dt=None):
     
    acc_results = [m[0].result() for m in acc_metrics]    
    batch_info = [epoch+1, n_epochs, batch+1, n_batches, time_str(t_taken), time_str(disp_time)]
    batch_info += [m[0].result() for m in loss_metrics] + acc_results
    end = '\n' if mean_dt else '\r'    
    print(report_line1.format(*batch_info), end=end)
    
    if mean_dt: # Test/validation
        acc_results = [m[1].result() for m in acc_metrics]
        batch_info = [mean_dt] + [m[1].result() for m in loss_metrics] + acc_results
        print(report_line2.format(*batch_info))
 

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
                           

def make_inference(model_path, profiles, replica_cols, klasses=None, batch_size=128,
                   ndim_compress=10, nlayers_att=2, nheads_att=2, n_rep=100):
    
    all_idx = range(len(profiles))
    data_generator = MixedLocReconstructDataGenerator(all_idx, profiles, klasses, replica_cols,
                                                      batch_size=batch_size, training=False, n_mix=0)
    
    n_classes = data_generator.n_classes
    n, p = profiles.shape
    out_array_r = np.zeros((n, n_rep, p), np.float32)
    out_array_c = np.zeros((n, n_rep, n_classes), np.float32)
    latent_array = np.zeros((n, n_rep, ndim_compress))
    
    model = get_model(data_generator, ndim_compress, nlayers_att, nheads_att, noise_sigma=0.0)
    model.load_weights(model_path)
 
    ref_profiles = data_generator.ref_profiles
    #ref_profiles = ops.convert_to_tensor(ref_profiles)

    n = len(all_idx)
    i = 0
    
    info(f'Making inference for {n:,} profiles')
    
    for batch, (x_in, y_true, weights) in enumerate(data_generator):
        info(f' .. {i:,}', end='\r')
        j = min(n, i+batch_size)
        
        for r in range(n_rep):
            y_pred_r, latent, y_pred_c = model([x_in, ref_profiles], training=False)
                         
            #cat_probs = np.exp(y_pred_c[:j-i]) # Convert from output log probs; softmax
            #cat_probs /= cat_probs.sum(axis=-1)[:,None]
            cat_probs = y_pred_c[:j-i]
            
            latent_array[i:j,r] = latent[:j-i]
            out_array_c[i:j,r] = cat_probs
            out_array_r[i:j,r] = y_pred_r[:j-i]

        i = j
    
    info('')

    return out_array_c, out_array_r, latent_array

"""
class layers.Dense(tfp.layers.layers.Dense):
    
    def __init__(self, parent, ndim, name, activation='gelu'):
        
        super().__init__(ndim, name=name, activation=activation)
        
        self.parent = parent
     

    def call(self, x):
        
        y = super().call(x)
	
        self.parent.add_loss(self.losses)

        return y



class BayesInference(keras.Layer):
    
    def __init__(self, ndim, name, activation='gelu'):
        
        super().__init__()
        
        self.sub_layer = layers.Dense(self, ndim, name=name, activation=activation)
 	    

    def call(self, x, training=None):

        return self.sub_layer(x)
"""

from .dv import DenseReparameterization


class CrossAttention(keras.Layer):

    def __init__(self,  n_heads, att_dim, ff_dim, dropout=0.1, activation='gelu', kernel_initializer='he_normal',
                 attention_mask=None, name='CA', *args, **kw):
        
        super().__init__(*args, **kw)
        
        self._attention_mask = attention_mask
        
        self._mha = layers.MultiHeadAttention(n_heads, att_dim, dropout=dropout)
        self._norm1 = layers.LayerNormalization()
        self._norm2 = layers.LayerNormalization()
        self._dens1 = layers.Dense(ff_dim, activation=activation,
                                   kernel_initializer=kernel_initializer)
        self._dens2 = layers.Dense(att_dim,) # Simple, linear        
        self._add1 = layers.Add()
        self._add2 = layers.Add()
        
    def call(self, query, key):
        
         x = self._mha(query, key, attention_mask=self._attention_mask, return_attention_scores=False)

         att_out = self._norm1(self._add1([x, query]))
 
         return self._norm2(self._add2([self._dens2(self._dens1(att_out)), att_out]))
         

def get_model(data_generator, ndim_compress=10, nlayers_att=4, nheads_att=2, noise_sigma=0.05):
    
    n_classes = data_generator.n_classes # One more than input markers/labels given on-the-fly null class

    n_ref, prof_dim =  data_generator.ref_profiles.shape[1:]
 
    x_query_in = keras.Input(shape=(prof_dim,), name='in_prof') # Queries, to augment. Sequence lengths is (1,) : batch will be (b, 1, p)
    x_keys_in  = keras.Input(shape=(n_ref, prof_dim), name='in_ref')  # Keys, to comare to, invariant over epoch : batch will be (1, n, p)

    compress_layer = layers.Conv1D(ndim_compress, 1, name='compress', activation='gelu', padding='valid')
 
    x1 = layers.Reshape((1,prof_dim), name='in_seq')(x_query_in) # From profiles to unarary sequence of profiles
    x1 = compress_layer(x1)
    x2 = compress_layer(x_keys_in)
 
    x1 = layers.GaussianNoise(noise_sigma, name=f'noise')(x1)
    x1 = layers.Dropout(0.1)(x1)
    
    x2 = ops.repeat(x2, data_generator.batch_size, axis=0) # Expand proteome ref profiles over minibatch 

    for i in range(nlayers_att):
        x1 = CrossAttention(nheads_att, ndim_compress, ndim_compress, name=f'MHAttention_{i+1}')(x1, x2)
   
    x1 = latent = layers.Reshape((ndim_compress,), name='latent')(x1)
    x1 = layers.LayerNormalization()(x1)

    # Reconstruction
    
    x1_recon = layers.Dense(ndim_compress, name='bnn_expand1', activation='gelu')(x1)
    x1_recon = layers.LayerNormalization()(x1_recon)
    
    recon_out = layers.Dense(prof_dim, name='recon_out', activation='sigmoid')(x1_recon)

    # Class mixtures

    x1_class = layers.Dense(ndim_compress, name='bnn_pred1', activation='gelu')(x1)
    x1_class = layers.LayerNormalization()(x1_class)
    
    x1_class = layers.Dense(ndim_compress, name='bnn_pred2', activation='gelu')(x1_class)
    x1_class = layers.LayerNormalization()(x1_class)
    
    class_mix_out = DenseReparameterization(n_classes, name='pred_out', activation='softmax')(x1_class)
 
    model = keras.Model([x_query_in, x_keys_in], [recon_out, latent, class_mix_out])
    
    return model
 
 
 
def train_model(model_path, test_idx, train_idx, profiles, class_labels, replica_cols, 
                n_mix=500, n_epochs=100, batch_size=32, init_learning_rate=1e-3,
                ndim_compress=10, nlayers_att=4, nheads_att=2):
  
    # Test and train on-the-fly data generators to create mixed profiles and classes
    data_generator = MixedLocReconstructDataGenerator(train_idx, profiles, class_labels, replica_cols, n_mix=n_mix)
    data_generator_test = MixedLocReconstructDataGenerator(test_idx, profiles, class_labels, replica_cols, training=False, n_mix=n_mix)

    model = get_model(data_generator, ndim_compress, nlayers_att, nheads_att)
    
    n_batches = data_generator.n_batches
    learning_rate = keras.optimizers.schedules.CosineDecay(init_learning_rate, n_epochs * n_batches, alpha=0.01)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    acc_funcs  = [mix_acc_mask, mae_acc, nonzero_mae, mae_mask]
   
    acc_metric_names  = ['acc_cat', 'acc_recon', 'mae_recon', 'mae_msk']
    loss_metric_names = ['loss_c','loss_r']

    loss_metrics, acc_metrics, report_line1, report_line2 = _init_metrics(loss_metric_names, acc_metric_names)    
    all_metrics = loss_metrics + acc_metrics
    

    def test_train_step(x_train, y_true, weights, training=True):
        x_in, x_ref = x_train
        y_true_c, y_true_r = y_true
        loss_metric_c, loss_metric_r = loss_metrics
 
        if training:
            with tf.GradientTape() as tape:
                y_pred_r, latent, y_pred_c = model(x_train, training=training)
                loss_c = neg_log_likelihood(y_true_c, y_pred_c, weights)
                loss_r = nonzero_mae_loss(y_true_r, y_pred_r, weights)
                loss_m = ops.sum(model.losses)/data_generator._n_items
                loss = loss_c + loss_r + loss_m
            
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
 
        else:
            y_pred_r, latent, y_pred_c = model(x_train, training=training)
            loss_c = neg_log_likelihood(y_true_c, y_pred_c, weights)
            loss_r = nonzero_mae_loss(y_true_r, y_pred_r, weights)
            loss = loss_c + loss_r 
 
        j = 0 if training else 1 # training, test
        loss_metric_c[j](loss_c)
        loss_metric_r[j](loss_r)

        for i, (acc_metric, acc_func) in enumerate(zip(acc_metrics, acc_funcs)):
            if i < 2:
                acc_metric[j](acc_func(y_true_c, y_pred_c, weights))
            else:
                acc_metric[j](acc_func(x_in, y_true_r, y_pred_r, weights))

    mean_dt = 0
    history = defaultdict(list)

    for epoch in range(n_epochs):
 
        for m1, m2 in all_metrics:
            m1.reset_state()
            m2.reset_state()
 
        n_steps = 0.0
        t_taken = 0.0
        t_prev = 0.0
        t_first = 0.0
        epoch_start = time.time()
 
        for batch, (x_train, y_true, weights) in enumerate(data_generator):
                        
            if batch == 0:
                epoch_start = time.time() # After generator initialisation 
                      
            ref_input = data_generator.ref_profiles
            test_train_step([x_train, ref_input], y_true, weights)
  
            n_steps += 1.0
            batch_end = time.time()
 
            if batch_end > (t_prev + REPORT_INTERVAL):
                t_taken = batch_end-epoch_start
 
                if t_first:
                    batch_time = (t_taken-t_first)/(n_steps-1.0)
                    disp_time = t_first + (n_batches-1) * (batch_time)
                else:
                    t_first = t_taken
                    batch_time = t_taken / n_steps
                    disp_time = n_batches * batch_time # Est total

                t_prev = batch_end                
            
                _report(epoch, n_epochs, batch, n_batches, 0.0, t_taken, disp_time,
                        loss_metrics, acc_metrics, report_line1, report_line2)
 
        t_taken = time.time()-epoch_start
        mean_dt = int(1e3 * t_taken/n_steps) # Miliseconds
        data_generator.on_epoch_end()
        
        # Test
        for x_test, y_true, weights in data_generator_test:
            test_train_step([x_test, ref_input], y_true, weights, training=False)
 
        v_time = time.time()-epoch_start
        v_time -= t_taken
        
        _report(epoch, n_epochs, batch, n_batches, v_time, t_taken, disp_time,
                loss_metrics, acc_metrics, report_line1, report_line2, mean_dt)
                
        data_generator_test.on_epoch_end()
 
        for m1, m2 in all_metrics:
            history[m1.name].append(m1.result())
            history[m2.name].append(m2.result())
 
    print('')
    info('Saving weights')
 
    model.save_weights(model_path)
 
    return history

 

