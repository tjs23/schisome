import math
import numpy as np
import keras
import tensorflow as tf

from scipy.spatial import distance    
from random import randint, random, shuffle, seed


class BaseGenerator(keras.utils.Sequence): # Superclass
  
    def __init__(self, iter_idx, profiles_in, batch_size, training, nan_val=0.0):
        
        #super().__init__(workers=1, use_multiprocessing=False, max_queue_size=10)
        
        self.profiles_in = np.nan_to_num(profiles_in, nan_val) # Everything
        self.iter_idx = list(iter_idx) # Subset of knowns for training/testing
        self._n_items = len(iter_idx)
        self.n_batches = int(math.ceil(self._n_items/batch_size))
        self.iterator = None
        self.batch_size = batch_size
        self.training = training
        self.epoch = 0
        
        self._set_reference()


    def _set_reference(self):
 
        # Reference array of all input profiles
 
        n, p = self.profiles_in.shape
        ref = np.array(self.profiles_in, dtype=np.float32).reshape(1, n, p)  # Will be broadcast to batches
        self.ref_profiles = ref # keras.ops.convert_to_tensor(ref)
 
 
    def make_iterator(self):
 
        pass # Overwite in subclass
 
 
    def __len__(self):
 
        return self.n_batches
 
 
    def __getitem__(self, idx):
        
        if idx == 0:
            self.iterator = self.make_iterator()
 
        data_items = next(self.iterator) 
    
        return data_items #x_inputs, y_true, weights    

    def on_epoch_end(self):
 
      self.batch = 0
      self.epoch += 1
      self._iterator = None
      

class SingleLocDataGenerator(BaseGenerator):
    
    def __init__(self, iter_idx, profiles_in, known_klasses, batch_size=32,
                             training=True):
        
        super().__init__(iter_idx, profiles_in, batch_size, training)
        
        self.known_klasses = known_klasses # array of class indices
                
        
    def make_iterator(self):
        
        profiles_in = self.profiles_in
        known = self.known_klasses
        
        k = known.max()+1
        prof_dim = self.ref_profiles.shape[2]
        b = self.batch_size
        tidx = self.iter_idx
        t = len(tidx)
        
        in_array    = np.zeros((b, 1, prof_dim), np.float32) # profiles
        out_array = np.zeros((b, k), np.float32) # subcell loc class, one-hot encoded
        weights     = np.empty((b, 1), np.float32) # Set to zero at batch end padding 
        
        i = 0
        
        while i < t-b:
            i_next = i + b # Start of next batch
            i_lim = min(i_next, t) # Limit of this batch useful data
            out_array.fill(0.0) # Mostly zeros in the end
            
            for batch_pos in range(b):
                j = i + batch_pos # Global index within current batch
                idx = tidx[j]
                 
                if j < i_lim:
                    weights[batch_pos,0] = 1.0
                    in_array[batch_pos,0,:] = profiles_in[idx]
                    out_array[batch_pos, known[idx]] = 1.0
                
                else: # Last batch padding
                    weights[batch_pos,0] = 0.0
                    in_array[batch_pos,0,:] = 0.0
                    
            yield in_array, out_array, weights
                
            i = i_next

    
class MixedLocReconstructDataGenerator(BaseGenerator):
    
    def __init__(self, iter_idx, profiles_in, known_klasses, replica_cols, batch_size=32,
                 training=True, mask_min=0.05, mask_max=0.25, mask_val=0.0, rep_mask=0.25,
                 n_mix=250, nan_val=0.0, max_null_corr=0.45):
        
        max_class = known_klasses.max()
        max_class += 1
        
        if training:
            n_prof, n_cols = profiles_in.shape
            n_null = 2 * n_mix # n_prof 
            null_profs = np.zeros((n_null, n_cols))
            profiles_in = np.concatenate([profiles_in, null_profs]) # Outputs for nulls always nan so not trainable, inputs filled every batch
            known_klasses = np.concatenate([known_klasses, np.full(n_null, max_class)])
        
        ni = len(iter_idx)
        
        if training:
            
            if n_mix:
                k = max_class + 1
                n_klass_pair = (k * k + k)//2
                ni += n_klass_pair * n_mix
                
            else:
                n_mix = 1
                     
        else:
            n_mix = 1    
        
        self.n_batches = int(math.ceil(ni/batch_size))
        self.n_classes = max_class + 1 # Add one for on-the-fly null class
        self.n_mix = n_mix
        self.known_klasses = known_klasses # array of class indices ; reserve 0 for unknown/fake/random
        self.mask_min = max(1, int(mask_min * profiles_in.shape[1]))
        self.mask_max = int(mask_max * profiles_in.shape[1])
        self.mask_val = mask_val
        self.nan_val = nan_val
        self.rep_mask = rep_mask
        self.replica_cols = replica_cols
        self.max_null_corr = max_null_corr
        self.ran_seed = randint(1, 2**31-1)
        
        super().__init__(iter_idx, profiles_in, batch_size, training, nan_val)
                
        
    def make_iterator(self):

        known = self.known_klasses
        profiles_in = self.profiles_in
        mask_min = self.mask_min
        mask_max = self.mask_max
        mask_val = self.mask_val
        iter_idx = self.iter_idx
        mix_idx    = [i for i in iter_idx if known[i] >= 0] # Negative is unknown
        n_mix = self.n_mix
        
        max_klass = known.max()
        
        if not self.training:
            max_klass += 1
        
        n, p = profiles_in.shape
        n_klass = max_klass+1
        n_rep = len(self.replica_cols) 
        r = n_rep-1
        rep_starts = np.cumsum(self.replica_cols)
                
        d, n, p = self.ref_profiles.shape
        b = self.batch_size
        
        # Subset of pairs (dense, classified random)
        if self.training:
            
            # set null class data, should not closely correlate with real data
            
            null_idx = (known == max_klass).nonzero()[0]
            n_replace = n_null = len(null_idx)
            n_orig = n - n_null
            
            while n_replace:
                
                ran_rows = np.random.randint(0, n_orig-1, (n_replace, max(self.replica_cols))) # Random real proteins, same for each replica
                ran_rows = np.concatenate([ran_rows[:,:rep_width] for rep_width in self.replica_cols], axis=1) # Clip to replica widths and join replicas

                ran_rows = ran_rows.ravel() # n_null * p
                ran_cols = np.concatenate([np.arange(p) for i in range(n_replace)])
                
                profiles_in[null_idx,:] = profiles_in[ran_rows, ran_cols].reshape(n_replace, p)
                corr_mat = 1.0 - distance.cdist(profiles_in[:n_orig], profiles_in[n_orig:], metric='correlation')
                max_corrs = corr_mat.max(axis=0)
                
                null_idx = n_orig + (max_corrs > self.max_null_corr).nonzero()[0]
                n_replace = len(null_idx)
                            
            pair_idx = []
            
            if n_mix > 1:
                klass_idx = {}
 
                for klass in range(n_klass):
                    klass_idx[klass] = (known == klass).nonzero()[0]

                # From each class to each other: get same num protein pairs for each dual class combo
                for i in range(0, n_klass):
                    kidx1 = klass_idx[i]
                    n1 = len(kidx1)
                    
                    if not n1:
                        print(f'Class {i} empty')
                        continue
 
                    for j in range(i,n_klass):
                        kidx2 = klass_idx[j]
                        n2 = len(kidx2)
                        
                        if not n2:
                            print(f'Class {j} empty')
                            continue
 
                        np.random.shuffle(kidx1)
                        np.random.shuffle(kidx2)
 
                        pair_idx += [(kidx1[q % n1], kidx2[q % n2]) for q in range(n_mix)]
     
            for idx1 in iter_idx:
                pair_idx += [(idx1, idx1)] # Original profile always present
            
            # Add null class"
                        
            shuffle(pair_idx)
                
        else:
            pair_idx = [(idx, idx) for idx in iter_idx]
        
        t = len(pair_idx)
        
        in_array = np.zeros((b, 1, p), np.float32) # masked profiles
        out_array_r = np.zeros((b, p), np.float32) # original profiles
        out_array_c = np.zeros((b, n_klass), np.float32) # subcell loc class, fractional prob encoding
        weights = np.empty((b, 1), np.float32) # Set to zero at batch end padding 
        
        i = 0
            
        seed(self.ran_seed) # Each epoch to have same randomm masking 
	
        while i < t:
            i_next = i + b # Start of next batch
            i_lim = min(i_next, t) # Limit of this batch useful data
            out_array_c.fill(0.0) # Mostly zeros in the end
            if self.training:
                self._set_reference()
           
              
            for batch_pos in range(b):
                j = i + batch_pos # Global index within current batch
                 
                if j < i_lim:
                    idx1, idx2 = pair_idx[j]
                    weights[batch_pos,0] = 1.0
                    f = random()
                    g = 1.0 - f
                    
                    row1 = np.array(profiles_in[idx1])
                    row2 = np.array(profiles_in[idx2])
                    mix_row = f*row1 + g*row2
                    mix_row[row1 == self.nan_val] = self.nan_val
                    mix_row[row2 == self.nan_val] = self.nan_val
                    klass1 = known[idx1]
                    klass2 = known[idx2]
 
                    out_array_r[batch_pos,:] = mix_row[:] # Out not masked
 
                    if self.training:
                        maskable = mix_row != self.nan_val # Not NaN

                        if random() < self.rep_mask: # Mask a whole replicate
                            rep = randint(0, r)
                            pos1 = rep_starts[rep]
                            pos2 = pos1 + self.replica_cols[rep]
                            maskable[:pos1] = False
                            maskable[pos2:] = False
                            mix_row[maskable] = mask_val
                            row1[maskable] = mask_val
                            row2[maskable] = mask_val
                            
                        else: # Mask a random scater of non-zeros
                            n_mask = randint(mask_min, mask_max)
                            maskable = maskable.nonzero()[0]
                            np.random.shuffle(maskable)
                            idx_mask = maskable[:n_mask]
                            mix_row[idx_mask] = mask_val
                            row1[idx_mask] = mask_val
                            row2[idx_mask] = mask_val
 
                        self.ref_profiles[0,idx1,:] = row1 # Stop short-cut via reference
                        self.ref_profiles[0,idx2,:] = row2
                    
                    in_array[batch_pos,0,:] = mix_row # Input maybe masked
                    
                    if (klass1 >= 0) and (klass2 >= 0): # Both defined; these add to 1.0 if klass1 == klass2
                        out_array_c[batch_pos, klass1] += f
                        out_array_c[batch_pos, klass2] += g
                 
                    
                else: # Last batch padding
                    weights[batch_pos,0] = 0.0
                    in_array[batch_pos,0,:] = 0.0
                    out_array_r[batch_pos,:] = 0.0
                            
            yield in_array, [out_array_c, out_array_r], weights
            
            i = i_next

