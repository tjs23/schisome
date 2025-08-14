import os
import numpy as np
from collections import defaultdict

from general_util import open_file, get_uniprot_columns, get_uniprot_alt_ids, get_color_array, info, warn

from constants import PROFILE_TAG, PROJ_TAG, RAW_DATA_TAG, ARRAY_VALUES_TAG, UNKNOWN, NAN
from constants import MARKER_CLASSES_TAG, MARKER_COLORS_TAG, MARKER_LABELS_TAG, PRED_CLASSES_TAG, PRED_LABELS_TAG

import warnings
warnings.filterwarnings("ignore")

class SchisomeException(Exception):
    
    def __init__(self, *args):
   
        super().__init__(args)


class BaseDataSet:
    """
    Base class for dataset handles all loading and saving of profile and markers (as NumPy zip)
    No plotting methods
    No DNN workflow methods
    """
    
    def __init__(self, file_path, aux_marker_key=None):
        
        file_root, file_ext = os.path.splitext(file_path)
            
        self._data_store = file_path
        self._temp_file = os.path.splitext(file_path)[0] + '__temp__.npz'
        
        if os.path.exists(file_path):
            if file_ext != '.npz':
                msg = 'File does not have .npz file extension'
                raise SchisomeException(msg)
                
            elif aux_marker_key:
                self._save_data({'aux_marker_key': aux_marker_key})
                
            else:
                save_dict = self._get_save_dict()
                aux_marker_key = save_dict.get('aux_marker_key')
                                            
        elif file_ext != '.npz':
            file_path = f'{file_root}.npz'        
         
        self._id_map = None
        self._raw_markers_key = 'organelle'
        self._train_markers_key = 'training'
        self._aux_markers_key = aux_marker_key
        
        self._train_profile_key = 'init'
        self._latent_profile_key = 'latent'
        self._recon_profile_key = 'recon'
        self._zfill_profile_key = 'zfill'
        
        self._pred_all_key = 'class_pred_all'                
        self._train_groups_key = 'train_groups'
    
    
    @property
    def has_predictions(self):
        
        return self.has_pred_class_key(self._pred_all_key) 
    

    @property
    def train_labels(self):
    
        return self.get_marker_labels(self._train_markers_key)


    @property
    def train_groups(self):
    
        return self.get_marker_data(self._train_groups_key)
         
    @property
    def train_groups_key(self):
    
        return self._train_groups_key
         
    
    @property
    def class_ensemble_preds(self):
         
        return self.get_pred_class_data(self._pred_all_key) 
 
 
    @property
    def class_ensemble_key(self):
         
        return self._pred_all_key
       
        
    @property
    def train_profile_key(self):
            
        return self._train_profile_key
    
    
    @property
    def train_profiles(self):
 
        return self.get_profile_data(self._train_profile_key)
 
 
    @property
    def latent_profile_key(self):
            
        return self._latent_profile_key
    
    
    @property
    def latent_profiles(self):
 
        return self.get_profile_data(self._latent_profile_key)
            
            
    @property
    def recon_profile_key(self):
            
        return self._recon_profile_key
    
    
    @property
    def recon_profiles(self):
 
        return self.get_profile_data(self._recon_profile_key)


    @property
    def zfill_profile_key(self):
            
        return self._zfill_profile_key
    
    
    @property
    def zfill_profiles(self):
 
        return self.get_profile_data(self._zfill_profile_key)
 
 
    @property
    def raw_markers_key(self):
    
        return self._raw_markers_key
        
        
    @property
    def raw_markers(self):
    
        return self.get_marker_data(self._raw_markers_key)


    @property
    def train_markers_key(self):
    
        return self._train_markers_key


    @property
    def train_markers(self):
    
        return self.get_marker_data(self._train_markers_key)
 
 
    @property
    def aux_markers(self):
    
        return self.get_marker_data(self._aux_markers_key)


    @property
    def aux_markers_key(self):
    
        return self._aux_markers_key
 
 
    def _check_file_path(self, file_path):
        
        if not os.path.exists(file_path):
            msg = f'File "{file_path}" doe not exist'
            raise SchisomeException(msg)
    
    
    def _check_pids(self, info='proceed'):
        
        save_dict = self._get_save_dict()
        
        if 'pids' not in save_dict:
            msg = f'Cannot {info}. No protein IDs loaded'
            raise SchisomeException(msg)
    
    
    def _get_save_dict(self):
        
        if os.path.exists(self._data_store):
            return np.load(self._data_store, allow_pickle=True, encoding='latin1')        
        
        else:
            return {}
        
            
    def _delete_save_data(self, key):
        
        save_dict = dict(self._get_save_dict())
        
        if key in save_dict:
            del save_dict[key]
            np.savez(self._temp_file, **save_dict)
            os.rename(self._temp_file, self._data_store)
     
        else:
            self.warn(f'Key {key} not present in "{self._data_store}"')
            
    
    def _save_data(self, new_data):
        
        save_dict = dict(self._get_save_dict())
        
        if save_dict:
            save_dict.update(new_data)
            np.savez(self._temp_file, **save_dict)
            os.rename(self._temp_file, self._data_store)
        
        else:
            self.info(f'Creating save file "{self._data_store}"')
            np.savez(self._data_store, **new_data)


    def _get_uniprot_cache(self, redo=False, cache_file='uniprot_cache.tsv'):

        pids = [pid.split('-')[0] for pid in self.proteins if not pid.startswith('cRAP')]
        up_dict = {}
        missing_in_cache = set()
        save_dict = self._get_save_dict()
        
        if cache_file and os.path.exists(cache_file):
            if redo:
                missing_in_cache = set(pids)
                self.info(f'Remaking cache {cache_file}')
            
            else:
                self.info(f'Reading cache {cache_file}')
                with open(cache_file) as file_obj:
                    head = file_obj.readline()
 
                    for line in file_obj:
                        pid, gene_name, description, xref_araport, xref_tair = line.strip('\n').split('\t')
                        up_dict[pid] = (gene_name, description, xref_araport, xref_tair)
 
                missing_in_cache = set([pid for pid in pids if pid not in up_dict])             
                             
        else:
            self.info(f'Cannot find cache {cache_file}')
            missing_in_cache = set(pids)
        
        self.info(f'Cache size {len(up_dict):,}')
             
        if missing_in_cache:
            self.info(f'Proteins missing in cache: {len(missing_in_cache):,}')
            up_dict2 = get_uniprot_columns(sorted(missing_in_cache), ['gene_primary', 'protein_name', 'xref_araport', 'xref_tair'])
            up_dict.update(up_dict2)
            
            missing_in_uniprot = []
            for pid in pids:
                pid = pid.split('-')[0]
                if pid not in up_dict:
                    missing_in_uniprot.append(pid)
            
            if missing_in_uniprot:
                self.info(f'Invalid UniProt IDs {len(missing_in_uniprot):,}')
                resolved = 0
 
                old_to_new = get_uniprot_alt_ids()
                obsolete_to_valid = {}
                new_ids = set()
 
                for pid in missing_in_uniprot:
                    if pid in old_to_new:
                        nid = old_to_new[pid]
                        obsolete_to_valid[pid] = nid
                        new_ids.add(nid)
                        resolved += 1
                 
                if new_ids:
                    up_dict2 = get_uniprot_columns(sorted(new_ids), ['gene_primary','protein_name', 'xref_araport', 'xref_tair'])
                    up_dict.update(up_dict2)
 
                    pids = [obsolete_to_valid.get(pid, pid) for pid in pids]
                    self._save_data({'pids':pids})

            with open(cache_file, 'w') as file_obj:
                line = 'protein_id\tgene_name\tdescription\txref_araport\txref_tair\n'
                file_obj.write(line)
                
                for protein_id in up_dict:
                     gene_name, description, xref_araport, xref_tair = up_dict[protein_id]
                     line = f'{protein_id}\t{gene_name}\t{description}\t{xref_araport}\t{xref_tair}\n'
                     file_obj.write(line)
        
            self.info(f'Updated {cache_file} with {len(up_dict):,} proteins')
        
        if ('alt_ids' not in save_dict) or missing_in_cache:
            alt_ids = []
            
            for pid in pids:
                pid = pid.split('-')[0]
                
                if pid in up_dict:
                    agis = set()
                    gene_name, description, xref_araport, xref_tair = up_dict[pid]
                    xref_araport = xref_araport.strip(';')
                    xref_tair = xref_tair.strip(';')
 
                    for agi in xref_araport.split(';'):
                        if agi:
                            agis.add(agi)

                    for agi in xref_tair.split(';'):
                        if agi:
                            agis.add(agi)
 
 
                    alt_ids.append(';'.join(sorted(agis)))
                
                else:
                    alt_ids.append('')
            
            self._save_data({'alt_ids':alt_ids})
        
        return up_dict


    def normalize_profiles(self, label, col_norm=False):
     
        profile_data = self.get_profile_data(label)
        #nan_mask = np.isnan(profile_data)
        
        n, m = profile_data.shape
 
        if col_norm:
            for k in range(m):
                col = profile_data[:,k]
                profile_data[:,k] /= np.median(col[col > 0])

        #div = np.nanmax(profile_data, axis=1) - np.nanmin(profile_data, axis=1)
        #nz = div != 0.0
        #profile_data[nz] /= div[nz,None]
        
        mx = np.nanmax(profile_data, axis=1)
        div = mx - np.nanmin(profile_data, axis=1)
        nz = div != 0.0
        profile_data[nz] /= mx[nz,None]
        
        self.set_profile_data(label, profile_data)
     
    
    def update_protein_ids(self):
        
        self._get_uniprot_cache(redo=True)
        
    
    def get_prot_titles(self, pids):

        title_dict = {}
        up_dict = self._get_uniprot_cache()
 
        for pid in pids:
            if pid in up_dict:
                gene_name, desc = up_dict[pid][:2]
            else:
                gene_name, desc = pid, 'Unknown'
        
            if ' (' in desc:
                desc = desc.split(' (')[0]
 
            title_dict[pid] = f'{gene_name or pid} : {desc or "Unknown"}'
 
        return title_dict
        
    
    @property
    def rev_id_map(self):
        
        self._get_uniprot_cache()
                
        rev_map = defaultdict(set)
        save_dict = self._get_save_dict()
        if 'alt_ids' in save_dict:
            pids = self.proteins
            alt_ids = save_dict['alt_ids']
            
            for pid, agis in zip(pids, alt_ids):
                for agi in agis.split(';'):
                    rev_map[pid].add(agi)

        return rev_map
            
    
    @property
    def id_map(self):
                
        if not self._id_map:
            self._get_uniprot_cache()
            self._id_map = defaultdict(set)
            
            save_dict = self._get_save_dict()
            
            if 'alt_ids' in save_dict:
                pids = self.proteins
                alt_ids = save_dict['alt_ids']
 
                for pid, aids in zip(pids, alt_ids):
                    for aid in aids.split(';'):
                        self._id_map[aid] = set([pid])
                    
        return self._id_map
            
            
    @property
    def marker_keys(self):
        
        self._check_pids('fetch marker list')
        marker_keys = []
        save_dict = self._get_save_dict()
        
        for key in save_dict:
            if key.startswith(MARKER_CLASSES_TAG):
                label = key[len(MARKER_CLASSES_TAG):]
                marker_keys.append(label)
        
        return marker_keys
        

    @property
    def pred_classes_labels(self):
        
        self._check_pids('fetch predicted classes list')
        clist = []
        save_dict = self._get_save_dict()
        
        for key in save_dict:
            if key.startswith(PRED_CLASSES_TAG):
                label = key[len(PRED_CLASSES_TAG):]
                clist.append(label)
        
        return clist


    def _have_label(self, save_tag, label):
        
        save_dict = self._get_save_dict()
        key = save_tag + label
        
        return key in save_dict
    
    
    def has_marker_key(self, label):
        
        return self._have_label(MARKER_CLASSES_TAG, label)


    def has_pred_class_key(self, label):
        
        return self._have_label(PRED_CLASSES_TAG, label)


    def _check_marker_label(self, label):
        
        if not self.has_marker_key(label):
            avail = ', '.join(self.marker_keys)
            msg = f'Marker set "{label}" not in marker list. Available: {avail}'
            raise SchisomeException(msg)
            
    
    def _check_pred_class_key(self, label):
    
        save_dict = self._get_save_dict()
        key = PRED_CLASSES_TAG + label
        
        if key not in save_dict:
            avail = ', '.join(self.pred_classes_labels)
            msg = f'Predicited classes "{label}" not in list. Available: {avail}'
            raise SchisomeException(msg)
    
    
    def get_marker_data(self, label):
    
        self._check_pids('fetch markers')
        self._check_marker_label(label)
        
        save_dict = self._get_save_dict()
        return save_dict[MARKER_CLASSES_TAG + label]

    
    def get_marker_labels(self, label):
        
        self._check_pids('fetch marker classes')
        self._check_marker_label(label)

        save_dict = self._get_save_dict()
        return [str(x) for x in save_dict[MARKER_LABELS_TAG + label]]


    def add_marker_proteins(self, marker_key, protein_ids, marker_labels, marker_colors=None):
 
        id_map = self.id_map
 
        pidx = {pid:i for i, pid in enumerate(self.proteins)}
 
        selection = np.zeros(len(pidx), int)
 
        for k, pids in enumerate(protein_ids):
            for pid in pids:
                if pid[:2] == 'AT':
                    uids = id_map.get(pid, pid)
                    for uid in uids:
                        selection[pidx[pid]] = k+1
                        break
                    
                elif pid in pidx:
                    selection[pidx[pid]] = k+1
 
        self.set_marker_data(marker_key, selection, marker_labels, marker_colors)
    
    
    def write_markers(self, key, out_file_path):

        labels = self.get_marker_labels(key)
        klasses = self.get_marker_data(key)
        pids = self.proteins
        n_written = 0
 
        label_dict = {i:x for i, x in enumerate(labels)}
        
        out_data = []
        
        for i, pid in enumerate(pids):
            k = klasses[i]
            
            if k < 0:
                continue
            
            klass = label_dict[k]
 
            if klass.lower() == 'unknown':
                continue
            out_data.append((klass, pid))
         
        out_data.sort()    
                
        with open(out_file_path, 'w') as out_file_obj:
            for klass, pid in out_data:
                line = f'{pid},{klass}\n'
                out_file_obj.write(line)
                n_written += 1
        
        self.info(f'Wrote {n_written:,} lines to {out_file_path}')
        
         
    def get_pred_class_data(self, key):
    
        self._check_pids('fetch predicted classes')
        self._check_pred_class_key(key)
        
        save_dict = self._get_save_dict()
        return save_dict[PRED_CLASSES_TAG + key]

    
    def get_pred_class_labels(self, key=None):
        
        if not key:
            key = self.train_markers_key
        
        self._check_pids('fetch predcited class labels')
        self._check_pred_class_key(key)

        save_dict = self._get_save_dict()
        return list(save_dict[PRED_LABELS_TAG + key])
        
        
    @property
    def marker_dict(self, key):
        
        marker_idx = list(self.get_marker_data(key))
        class_labels = list(self._get_save_dict()[MARKER_LABELS_TAG + key])
        
        mdict = {}
        for i, pid in enumerate(self.proteins):
            mdict[pid] = class_labels[marker_idx[i]]
        
        return mdict        
    
    @property
    def proteins(self):
        
        save_dict = self._get_save_dict()
        return list(save_dict.get('pids', []))
    
    
    @property
    def alt_ids(self):
        
        save_dict = self._get_save_dict()
        return list(save_dict.get('aids', []))
 
    
    def warn(self, msg, *args, **kw):
    
        warn(msg, *args, **kw)
    
    
    def info(self, msg, *args, **kw):
    
        info(msg, *args, **kw)
                
        
    def _load_profile_file(self, file_path):
        
        self._check_file_path(file_path)
                
        profile_dict = {}

        with open_file(file_path) as file_obj:
            #head1 = file_obj.readline()
 
            for line in file_obj:
                data = line.rstrip().split('\t')

                if data:
                    pid = data[0]
                    if pid.startswith('cRAP'):
                        continue
            
                    pid = pid.split(';')[0]
                    pid = pid.split('-')[0]
                    
                    prof = data[1:]
                    prof = [float(v or 'nan') for v in prof]
                    intens = sum(prof)
 
                    if intens:
                        profile_dict[pid] = prof
        
        return profile_dict    
    
    
    def add_raw_profiles(self, file_paths):
        
        if self.proteins:
            self.warn(f'Replacing raw profiles in {self._data_store}')
        
        pids = set()
        prof_dicts = []
        
        for i, file_path in enumerate(file_paths):
            replica = i + 1
            prof_dict = self._load_profile_file(file_path)
            pids.update(prof_dict)
            prof_dicts.append(prof_dict)
            n_frac = len(next(iter(prof_dict.values())))
            self.info(f'Loaded replicate {replica} data for {len(prof_dict):,} proteins covering {n_frac} columns/fractions from file_path')
     
        pids = sorted(pids)

        n = len(pids)
        n_frac_tot = 0
        new_data = {'pids': np.array(pids),}
        comb_profiles = []
        
        for i, file_path in enumerate(file_paths):
            prof_dict = prof_dicts[i]
            n_frac = len(next(iter(prof_dict.values())))
            n_frac_tot += n_frac
            data_array = np.empty((n, n_frac))
            
            for j, pid in enumerate(pids):
                if pid in prof_dict:
                    data_array[j] = prof_dict[pid]
                else:
                    data_array[j] = NAN
                
            label = f'{RAW_DATA_TAG}{i:04d}_{os.path.basename(file_path)}'
            new_data[label] = data_array
            comb_profiles.append(data_array)
        
        comb_profiles = np.concatenate(comb_profiles, axis=1)
        n, m = comb_profiles.shape

        new_data[PROFILE_TAG + self._train_profile_key] = comb_profiles
        self._save_data(new_data)
                
        self.info(f'Overall loaded {n:,} proteins covering {m} columns/fractions') 
    
    
    @property
    def profile_keys(self):
        
        self._check_pids('fetch profile keys')
        profile_keys = []
        save_dict = self._get_save_dict()
        
        for key in save_dict:
            if key.startswith(PROFILE_TAG):
                label = key[len(PROFILE_TAG):]
                profile_keys.append(label)
        
        return profile_keys

    
    @property
    def array_keys(self):
        
        self._check_pids('fetch array keys')
        array_keys = []
        save_dict = self._get_save_dict()
        
        for key in save_dict:
            if key.startswith(ARRAY_VALUES_TAG):
                label = key[len(ARRAY_VALUES_TAG):]
                array_keys.append(label)
        
        return array_keys
        
    
    def restore_original_profiles(self):
    
        if not self.proteins:
            self.warn('No original profile data to restore')
            return
     
        comb_profiles = []
        save_dict = self._get_save_dict()
        
        for key in save_dict:
            if key.startswith(RAW_DATA_TAG):
                comb_profiles.append(save_dict[key])
                
        comb_profiles = np.concatenate(comb_profiles, axis=1)
        n, m = comb_profiles.shape
        
        new_data = {PROFILE_TAG + 'initial': comb_profiles}
        self._save_data(new_data)
        
        self.info(f'Restored data for {n} proteins covering {m} columns/fractions') 
        
    
    def set_marker_data(self, label, marker_idx, marker_klasses, marker_colors=None):
        
        self._check_pids('set marker data')
        
        pids = self.proteins
        n = len(pids)

        if marker_idx.ndim != 1:
            msg = f'Marker indexes are not a one dimensional array'
            raise SchisomeException(msg)
        
        if n != len(marker_idx):
            msg = f'Marker index array length {len(marker_idx):,} does not match the number of proteins {n:,}'
            raise SchisomeException(msg)            
 
        new_data = {MARKER_LABELS_TAG + label: np.array(marker_klasses),
                    MARKER_CLASSES_TAG + label: marker_idx}
                                
        if marker_colors:
            new_data[MARKER_COLORS_TAG + label] = get_color_array(marker_colors)
                                
        self._save_data(new_data)
    
    
    def get_marker_color_dict(self, label):
    
        save_dict = self._get_save_dict()
        
        key1 = MARKER_LABELS_TAG+label
        key2 = MARKER_COLORS_TAG+label
        
        if key2 in save_dict:
            color_dict = dict(zip(save_dict[key1], save_dict[key2]))
            return color_dict
        
        else:
            return None


    def get_array_data(self, label):
    
        self._check_pids('fetch array data')
        self._check_marker_label(label)
        
        save_dict = self._get_save_dict()
        return save_dict[ARRAY_VALUES_TAG + label]
   
            
    def set_array_data(self, label, values):
        
        self._check_pids('set array data')
        
        pids = self.proteins
        n = len(pids)

        if values.ndim != 1:
            msg = f'Values are not a one dimensional array'
            raise SchisomeException(msg)
        
        if n != len(values):
            msg = f'Value array length {len(values):,} does not match the number of proteins {n:,}'
            raise SchisomeException(msg)            
 
        new_data = {ARRAY_VALUES_TAG + label: values}
                                
        self._save_data(new_data)


    def set_pred_class_data(self, label, klass_data, klass_labels):
        
        self._check_pids('set predicted class data')
        
        pids = self.proteins
        n = len(pids)

        if klass_data.ndim not in (2,3):
            msg = f'Predicted class data not a two or three dimensional array (n_proteins, [n_replicas], n_classes)'
            raise SchisomeException(msg)
        
        if n != len(klass_data):
            msg = f'Predicted class array length {len(klass_data):,} does not match the number of proteins {n:,}'
            raise SchisomeException(msg)            

        new_data = {PRED_LABELS_TAG + label: np.array(klass_labels),
                    PRED_CLASSES_TAG + label: klass_data}
        self._save_data(new_data)


    def set_profile_data(self, label, data_array):
        
        self._check_pids('set profile data')
        
        pids = self.proteins
        n = len(pids)

        if data_array.ndim != 2:
            msg = f'Profile data is not two dimensional (proteins rows, fraction columns)'
            raise SchisomeException(msg)
        
        if n != len(data_array):
            msg = f'Profile data length {len(data_array):,} does not match the number of proteins {n:,}'
            raise SchisomeException(msg)            

        key = PROFILE_TAG + label
        
        new_data = {key: data_array}
        self._save_data(new_data)
        
        save_dict = self._get_save_dict()
        
        for key in save_dict:
            if key.startswith(PROJ_TAG) and key.endswith(label):
                self.info(f'Removing {key}')
                self._delete_save_data(key)
        

    def have_profile_label(self, label):
        
        return self._have_label(PROFILE_TAG, label)


    def _check_profile_label(self, label):
    
        if not self.have_profile_label(label):
            avail = ', '.join(self.profile_keys)
            msg = f'Profile set "{label}" not in profiles list. Available: {avail}'
            raise SchisomeException(msg)
         
    
    def get_profile_data(self, label):
        
        self._check_profile_label(label)

        return self._get_save_dict()[PROFILE_TAG + label]
    
    
    def set_2d_proj(self, proj_2d, method, label):
     
        key = f'{PROJ_TAG}{method}_{label}'
        new_data = {key: proj_2d}
        self._save_data(new_data)

        return self._get_save_dict()[key] # Reload after change
    

    def get_2d_proj(self, method, label):
    
        save_dict = self._get_save_dict()
        
        key = f'{PROJ_TAG}{method}_{label}'
        
        if key in save_dict:
            return save_dict[key]
         
    
    def reset_2d_proj(self):
        
        save_dict = self._get_save_dict()
        
        for key in save_dict:
            if key.startswith(PROJ_TAG):
                self._delete_save_data(key)
                
    def add_markers(self, label, file_path):
    
        self._check_pids('add markers')
        
        if label in self.marker_keys:
            self.warn(f'Replacing marker set {label}')
         
        id_mapping = self.id_map
        
        pids = set(self.proteins)
        klass_dict = {}
        unknown = set()
        
        with open_file(file_path) as file_obj:
            head1 = file_obj.readline()
 
            for line in file_obj:
                line = line.rstrip()
                if not line:
                    continue
                
                data = line.split(',')
                agi, klass = data
                agi = agi.upper()
                klass = klass.upper()
                
                if klass == 'UNKNOWN':
                    continue
                
                if agi.startswith('AT'):
                
                    if agi not in id_mapping:
                        agi = agi.split('.')[0]
 
                        if agi not in id_mapping:
                            agi = agi + '.1'
                            
                            if agi not in id_mapping:
                                unknown.add(agi)
                                continue
 
                    pid = list(id_mapping[agi])[0]
                
                else:
                    pid = agi
                    pid = pid.split(';')[0]
                    pid = pid.split('-')[0]
                    
                
                klass_dict[pid] = klass
        
        missing = set(klass_dict) - pids
        
        if unknown:
            example = sorted(unknown)
            if len(example) > 5:
                example = example[:5] + ['...',]
                
            example = ', '.join(example)    
            self.warn(f'Cannot match {len(unknown):,} marker protein IDs to UniProt IDs. {example}')

        if missing:
            self.warn(f'A total of {len(missing):,} marker protein IDs in "{file_path}" are not found in the main dataset.')
            print(sorted(missing))
        
        klasses = list(set(klass_dict.values()))
        klass_idx = {k:i for i, k in enumerate(klasses)}
        
        n = len(self.proteins)
        marker_idx = np.empty((n,), dtype=int)
        
        n_added = 0
        
        for i, pid in enumerate(self.proteins):
            if pid in klass_dict:
                k = klass_idx[klass_dict[pid]]
                n_added += 1
            else:
                k = -1
                
            marker_idx[i] = k
        
        new_data = {MARKER_LABELS_TAG + label: np.array(klasses),
                                MARKER_CLASSES_TAG + label: marker_idx}
        
        self._save_data(new_data)

        self.info(f'Added markers "{label}": {n_added:,} proteins in {len(klasses)} classes')

                
    @property
    def replica_cols(self):
        
        self._check_pids('fetch replica columns')
        save_dict = self._get_save_dict()
        replica_widths = []
        
        keys = sorted([k for k in save_dict if k.startswith(RAW_DATA_TAG)])
         
        for key in keys:
            n, m = save_dict[key].shape
            replica_widths.append(m)
  
        
        if len(replica_widths) == 1:
            s = replica_widths[0]
            for w in (11, 18, 16, 10): # Commoon TMT tag counts
                if s % w == 0:
                    replica_widths = [w]  * (s//w)
                    break                     
        
        return replica_widths


    def get_valid_mask(self, min_nonzero=0.8, profile_key=None):
        
        if not profile_key:
            profile_key = self.train_profile_key
        
        prof_data = self.get_profile_data(profile_key)
        valid_mask = np.count_nonzero(np.nan_to_num(prof_data), axis=1) >= int(min_nonzero * prof_data.shape[1])
 
        return valid_mask


                         
    def write_profile_tsv(self, out_file_path, profiles=None, markers=None, write_blank=False):
        
        if markers is True:
            markers = self.marker_keys

        if not profiles:
            profiles = self.profile_keys
        
        save_dict = self._get_save_dict()
        n_lines = 0
        header = None
        
        if markers:
            marker_dicts = {m:{} for m in markers}
        
        prof_dicts = {p:{} for p in profiles}
        
        for key in sorted(save_dict):
            
            if key.startswith(MARKER_CLASSES_TAG):
                label = key[len(MARKER_CLASSES_TAG):]
         
                if markers and label not in markers:
                    continue
                
                key2 = MARKER_LABELS_TAG + label
                klasses = list(save_dict[key2])
                
                marker_data = save_dict[key]
                marker_dict = marker_dicts[label]
                
                for i, pid in enumerate(self.proteins):
                    k = int(marker_data[i])
                    
                    if k < 0:
                        klass = UNKNOWN
                    else: 
                        klass = klasses[k]
                    
                    marker_dict[pid] = klass
                
            elif key.startswith(PROFILE_TAG):
                label = key[len(PROFILE_TAG):]
                
                if profiles and label not in profiles:
                    continue
                
                prof_data = save_dict[key]
                prof_dict = prof_dicts[label]
                
                for i, pid in enumerate(self.proteins):
                    row = prof_data[i]
                    prof_dict[pid] = row
                    
        
        n_profs = len(prof_dicts)
        with open_file(out_file_path, 'w') as out_file_obj:
            write = out_file_obj.write
            header = ['protein_ID'] + list(markers)
            
            for pid in self.proteins:
                for label in profiles:
                    p = len(prof_dicts[label][pid])
                    
                    for i in range(p):
                        header.append(f'{label}_F{i+1}')
                    
                break
            
            line = '\t'.join(header) + '\n'
            write(line)
            
            for i, pid in enumerate(self.proteins):
                prof_vals = []
                n_blank = 0
                
                
                for label in profiles: # ordered
                    prof_dict = prof_dicts[label]
                    row = prof_dict[pid]
                    prof_vals += ['%.7f' % x for x in row]
                    
                    if not write_blank:
                        if np.all(np.isnan(row)):
                            n_blank += 1

                        if np.all(row == 0.0):
                            n_blank += 1
                
                if not write_blank and (n_blank == n_profs):
                    continue
                
                marker_klasses = []
                
                if markers:
                    for label in markers: # ordered
                        marker_dict = marker_dicts[label]

                        if pid in marker_dict:
                            marker_klasses.append(marker_dict[pid])
                        else:
                            marker_klasses.append(UNKNOWN)
                
                row = [pid] + marker_klasses + prof_vals
                line = '\t'.join(row) + '\n'
                write(line)
                n_lines += 1
        
        self.info(f'Wrote {n_lines:,} lines to {out_file_path}')
    
    
    def add_uniprot_markers(self, label, cache_path='markers/UniProt_Human_cc_subcellular_location.tsv'):
    
        from collections import defaultdict
        
        self._check_pids('add markers')
        
        if label in self.marker_keys:
            self.warn(f'Replacing marker set {label}')
        
        loc_dict = {}
        
        if os.path.exists(cache_path):
            with open(cache_path) as file_obj:
                for line in file_obj:
                    pid, cc_subcellular_location, reviewed = line.strip('\n').split('\t')
                    loc_dict[pid] = (cc_subcellular_location, reviewed)

        missing = [pid for pid in self.proteins if pid not in loc_dict]
        
        if missing:
            self.warn(f'Missing UniProt cache info for {len(missing):,}')
            loc_dict.update(get_uniprot_columns(missing, ('cc_subcellular_location','reviewed')))
        
        with open(cache_path, 'w') as out_file_obj:
            for pid in sorted(loc_dict):
                cc_subcellular_location, reviewed = loc_dict[pid]
                out_file_obj.write(f'{pid}\t{cc_subcellular_location}\t{reviewed}\n')
                
        pids = set(self.proteins)
        
        n = len(self.proteins)
        marker_idx = np.empty((n,), dtype=int)
        klass_dict = {}
        
        for pid in loc_dict:
            if pid not in pids:
                #self.warn(f'Cannot lookup UniProt ID {pid}')
                continue

            scl, is_reviewed = loc_dict[pid]
            
            if is_reviewed != 'reviewed':
                continue
            
            #print(scl)
            scl = scl.replace('SUBCELLULAR LOCATION: ','')
            scl = scl.split(' Note=')[0]
             
            scls = scl.split('. ')
            scls = set([x.split(';')[0].split(' {')[0].split(',')[0].strip('.') for x in scls])
            scls = set([x.split(']: ')[1] if (']: ' in x) else x for x in scls])
            
            filtered = set()
            valid = set(['Cytoplasm', 'ER', 'Endosome', 'Extracellular', 'Golgi', 'Lysosome', 'Mitochondrion', 'Nucleus',    'PM', 'Peroxisome', 'Vacuole'])
                        
            for scl in scls:
                if scl.startswith('Golgi '):
                    scl = 'Golgi'
                    
                elif scl.startswith('Lysosome '):
                    scl = 'Lysosome'

                elif scl.startswith('Vacuole '):
                    scl = 'Vacuole'

                elif scl.startswith('Mitochondrion '):
                    scl = 'Mitochondrion'

                elif scl.startswith('Cytoplasmic '):
                    scl = 'Cytoplasm'

                elif scl.startswith('Nucleus '):
                    scl = 'Nucleus'

                elif scl.startswith('Extracellular '):
                    scl = 'Extracellular'
 
                elif scl.startswith('Endosome '):
                    scl = 'Endosome'

                elif scl.startswith('Chromosome'):
                    scl = 'Nucleus'
                    
                elif scl.startswith('Peroxisome '):
                    scl = 'Peroxisome'
                    
                elif 'cell membrane' in scl:
                    scl = 'PM'

                elif 'plasmic reticulum' in scl:
                    scl = 'ER'

                elif 'Secreted' in scl:
                    scl = 'Extracellular'

                elif 'endosome' in scl:
                    scl = 'Endosome'

                elif 'Microsome' in scl:
                    scl = 'ER'
                
                elif scl in ('Cell junction', 'Cell membrane', 'Cell projection', 'Cell surface'):
                    scl = 'PM'

                elif 'Melanosome' in scl:
                    continue
                
                if scl not in valid:
                    continue
                
                if scl == 'Cytoplasm':
                    scl = 'Cytosol'

                if scl == 'Mitochondrion':
                    scl = 'Mitochondria'
                 
                scl = scl.upper()
                
                filtered.add(scl)
        
            if len(filtered) != 1:
                continue
            
            klass = filtered.pop() 
            klass_dict[pid] = klass           
             
        klasses = sorted(set(klass_dict.values()))
        klass_idx = {k:i for i, k in enumerate(klasses)}
     
        n_added = 0
        for i, pid in enumerate(self.proteins):
            if pid in klass_dict:
                k = klass_idx[klass_dict[pid]]
                n_added += 1
            else:
                k = -1
                
            marker_idx[i] = k
        
        new_data = {MARKER_LABELS_TAG + label: np.array(klasses),
                                MARKER_CLASSES_TAG + label: marker_idx}
        
        self._save_data(new_data)

        self.info(f'Added markers "{label}": {n_added:,} proteins in {len(klasses)} classes')
