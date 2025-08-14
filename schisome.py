import os, math, colorsys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ColorConverter, LinearSegmentedColormap
from matplotlib import cm
from scipy import stats

from constants import COLOR_DICT, DB_ORGANELLE_CONV_DICT, DB_ORGANELLE_INFO
from sql_schema import DB_SCHEME
from base_dataset import BaseDataSet, SchisomeException


class SchisomeDataSet(BaseDataSet):
    """
    This class extends the BaseDataset with DNN, plotting and analysis methods
    """    
    
    def __init__(self, file_path, source_tag=None, aux_marker_key=None):
        
        super().__init__(file_path, aux_marker_key)
                
        if source_tag:
            self._save_data({'source_tag':source_tag})
            
        else:
            save_dict = self._get_save_dict()
            source_tag = save_dict.get('source_tag')

        if not source_tag:
            msg = 'File for dataset does not contain "source_tag" and none was specified'
            raise SchisomeException(msg)
         
        self.source_tag = source_tag
    
    
    def _plot_proj_2d(self, profile_labels, class_labels, titles, proj_methods,
                      title=None, save_path=None, min_nonzero=0.8, spot_size=10,
                      color_dict=None, label_ids=None):
        
        if isinstance(profile_labels, str):
            profile_labels = [profile_labels]

        if isinstance(class_labels, str):
            class_labels = [class_labels]
     
        n_proj = max(len(proj_methods), len(class_labels), len(profile_labels))

        if len(proj_methods) < n_proj:
            proj_methods = proj_methods * n_proj
            proj_methods = proj_methods[:n_proj]
     
        if len(profile_labels) < n_proj:
            profile_labels = profile_labels * n_proj
            profile_labels = profile_labels[:n_proj]

        if len(class_labels) < n_proj:
            class_labels = class_labels * n_proj
            class_labels = class_labels[:n_proj]
                         
        if not titles:
            titles = class_labels[:]
        
        elif len(titles) < n_proj:
            titles = titles * n_proj
            titles = titles[:n_proj]
                    
        for profile_label in profile_labels:
            self._check_profile_label(profile_label)
        
        marker_labels = self.marker_keys
        for class_label in class_labels:
            if class_label in marker_labels:
                self._check_marker_label(class_label)
            
            else:
                self._check_pred_class_key(class_label)
        
        self.plot_proj_2d(profile_labels, class_labels, proj_methods, titles,
                          title=title, min_nonzero=min_nonzero, save_file=save_path,
                          color_dict=color_dict, spot_size=spot_size, label_ids=label_ids)    
        

    def _plot_single_scatter_2d(self, ax, profile_label, class_label, proj_method, title,
                                size, min_nonzero=0.8, color_dict=None, label_ids=None, duals=False):

        profile_data = self.get_profile_data(profile_label)
        profile_data = np.nan_to_num(profile_data)
 
        if class_label in self.pred_classes_labels:
            score_array = self.get_pred_class_data(class_label)
            sort_scores = np.sort(score_array, axis=-1)
            invalid = (sort_scores[:,-1] - sort_scores[:,-2]) < 0.5

            class_array = score_array.argmax(axis=-1)
            class_array[invalid] = 0
            class_labels =    self.get_pred_class_labels(class_label) + ['unknown']
 
            if duals:
                duals = (sort_scores[:,-2:].sum(axis=-1) > 0.9) & (score_array.max(axis=-1) < 0.7)
                class_array[duals] = class_array.max() + 1
                class_labels.append('DUAL')
 
        else:
            class_array = self.get_marker_data(class_label)
            class_labels = self.get_marker_labels(class_label)
            class_array[class_array < 0] = len(class_labels)
            class_labels = class_labels + ['unknown']
            class_array[class_array >= len(class_labels)] = 0
 
        n, m = profile_data.shape
        valid = self.get_valid_mask(min_nonzero)
 
        select_idx = np.zeros(n, int)
        proteins = self.proteins
        if label_ids:
            # Selection always shown
            pidx = {x:i for i, x in enumerate(proteins)}
            selection = dict(label_ids)
 
            for pid in selection:
                if pid in pidx:
                    j = pidx[pid]
                    select_idx[j] = 1
                    valid[j] = True
 
        proj_2d = self.get_profile_proj_2d(profile_label, proj_method)
 
        x_vals, y_vals = proj_2d.T
        x_vals = x_vals[valid]
        y_vals = y_vals[valid]
        select_idx = select_idx[valid]
        class_array = class_array[valid]

        cmap = cm.get_cmap('afmhot')
        n_points = len(x_vals)
        n_marked = 0
 
        if color_dict:
            color_dict.update(COLOR_DICT)
        else:
            color_dict = self.get_marker_color_dict(class_label) or COLOR_DICT
 
        if label_ids:
            valid_ids = [pid for i, pid in enumerate(self.proteins) if valid[i]]
            for i, pid in enumerate(valid_ids):
                 if not select_idx[i]:
                     continue
 
                 if np.isnan(x_vals[i]) or np.isnan(y_vals[i]):
                     self.warn(f'No 2D projection location for {pid} : {selection[pid]}')
                     continue
 
                 txt = selection[pid]
                 ax.text(x_vals[i], y_vals[i], txt, fontsize=5, color='#B0B0B0', zorder=0)
 
        for i, klass in enumerate(class_labels):
 
            idx = (class_array == i).nonzero()[0]
 
            if klass in color_dict:
                color = color_dict[klass]
 
            else:
                color = cmap(i/float(len(class_labels)))

            if len(idx):

                if klass in ('unknown', 'DUAL', 'no-TM', 'UNKNOWN'):
                    alpha = 0.2
                    zorder = 0
 
                    if klass == 'DUAL':
                        marker = '*'
                    else:
                        marker = '.'
 
                else:
                    alpha = 0.75
                    zorder = 1+i
                    marker = '.'
                    n_marked += len(idx)
 
                label = klass.replace('OUTER MEMBRANE','OM',)
                label = label.replace('INNER MEMBRANE','IM',)
                label = label.replace('ENVELOPE','ENV',)
 
                if label_ids:
                    mask = select_idx[idx] == 0
                    ax.scatter(x_vals[idx][mask], y_vals[idx][mask], s=size*0.5, alpha=0.5, marker=marker, zorder=1, color=color)

                    mask = select_idx[idx] == 1
                    ax.scatter(x_vals[idx][mask], y_vals[idx][mask], s=size, alpha=1.0, marker=marker, zorder=i+1, color=color, edgecolors='w', linewidth=0.5)
                    ax.scatter([], [], s=size, alpha=1.0, marker=marker, color=color, label=label)
 
                else:
                    ax.scatter(x_vals[idx], y_vals[idx], s=size, alpha=alpha, marker=marker, zorder=zorder, label=label, color=color)
 
 
        ax.set_title(f'{title} {n_marked:,}/{n_points:,}')
    
        return profile_data, (x_vals, y_vals)

    
    
    def plot_proj_2d(self, profile_labels, marker_labels, proj_methods=['umap'], titles=None,
                     title=None, min_nonzero=0.8, plot_size=8.0, spot_size=10, alpha=0.25,
                     save_file=None, color_dict=None, label_ids=None):
         
        title_dict = {'umap':'UMAP','p-umap':'parametric-UMAP','pca':'PCA','tsne':'t-SNE',}
 
        if not title:
            title = '2D profile maps'
 
        while len(proj_methods) < len(profile_labels):
            proj_methods.append(proj_methods[-1])
 
        while len(marker_labels) < len(profile_labels):
            marker_labels.append(marker_labels[-1])
 
        if not titles:
            titles = [title_dict[method] for method in proj_methods]
 
        n_cols = len(proj_methods)
 
        fig, axarr = plt.subplots(1, n_cols, squeeze=False)
        fig.set_size_inches(n_cols * plot_size, plot_size)

        for col, profile_label in enumerate(profile_labels):
            marker_label = marker_labels[col]
            proj_method = proj_methods[col]
 
            self._plot_single_scatter_2d(axarr[0,col], profile_label, marker_label,
                                         proj_method, titles[col], spot_size, min_nonzero,
                                         color_dict, label_ids)
 
            axarr[0,col].legend(fontsize=7) # , loc='upper left')
 
        fig.suptitle('{}'.format(title), fontsize=16)
 
        plt.subplots_adjust(left=0.03, bottom=0.05, right=0.95, top=0.92, wspace=0.1, hspace=0.1)
 
        if save_file:
            plt.savefig(save_file, dpi=400)
            plt.clf()
            self.info(f'Saved image to {save_file}')
 
        else:
            plt.show()
 
        return plt

    
    def _print_confusion_matrix(self, known, predicted):

        self.info('Confusion matrix')
        
        m = predicted.max()
 
        out_map = np.zeros((m+1, m+1), int)
 
        for i, k1 in enumerate(predicted):
            k2 = known[i]
            if k2 >= 0:
                out_map[k1, k2] += 1
 
        self.info(out_map) # Check pred classes match marker indices
    
    
    def make_class_predictions(self, fit=True, fit_distrib_points=200, rv=stats.beta, single_thresh=0.8):
        
        def _ensemble_survival(rv, sample_values, distrib_points, group_step=100):
 
            n = sample_values.shape[0]
            cdf = None
            n_comp = 0.0
            for a in range(0, n, group_step):
 
                try:
                    params = rv.fit(sample_values[a:a+group_step], floc=0.0, fscale=1.0)
                except Exception as err:
                    continue
 
                if cdf is None:
                    cdf = rv.cdf(distrib_points, *params)
                else:
                    cdf += rv.cdf(distrib_points, *params)
 
                n_comp += 1.0
            
            if n_comp:
                cdf /= n_comp
            else:
                cdf = np.full(distrib_points.shape, 1.0/len(distrib_points))
                
            return 1.0 - cdf        
        
        def _ensemble_pdfs(rv, sample_values, distrib_points, group_step=100):
 
            n = sample_values.shape[0]
            b = distrib_points[1]
            spike =    1.0/b
            pdfs = []
            
            for a in range(0, n, group_step):
                vals = sample_values[a:a+group_step]
                
                if vals.max() < b: # Spike at zero, no point fitting
                    pdf = np.zeros(distrib_points.shape)
                    pdf[0] = spike
                
                else:
                    try:
                        params = rv.fit(vals, floc=0.0, fscale=1.0)
                        pdf = rv.pdf(distrib_points, *params)
 
                    except Exception as err:
                        pdf = np.zeros(distrib_points.shape)
                        pdf[0] = spike
                
                if not pdf.max():
                    pdf[0] = spike
                     
                pdfs.append(pdf)
            
            return pdfs

        self.info(f'Setting predictions, fitting p-values etc.')
        
        klass_labels = self.train_labels
        pred_class_scores = self.class_ensemble_preds
        
        n, m, nk = pred_class_scores.shape
 
        distrib_points = np.linspace(0.0, 1.0, fit_distrib_points)
 
        p_values1 = np.zeros(n) # p-value for a single prediction
        p_values2 = np.zeros(n) # p-value for a double prediction 
        singularity = np.zeros(n) # singleness scores
        duality = np.zeros(n) # dualness scores

        prediction = np.full(n, -1)
        prediction2 = np.full(n, -1)

        for i in range(n):
            prot_scores = pred_class_scores[i]
            mean_scores = prot_scores.mean(axis=0)
            best_first = mean_scores.argsort()[::-1]
 
            # Single, dual
 
            k1, k2 = best_first[:2]
            mv1 = mean_scores[k1]
            mv2 = mean_scores[k2]
            
            pdfs = np.array([np.array(_ensemble_pdfs(rv, prot_scores[:,j], distrib_points)) for j in range(nk)])
            t = int(mv1*fit_distrib_points) # threshold index
            t2 = int((mv1+mv2)*fit_distrib_points)
            
            # k x m x p -> k x p
            pdfs = pdfs.mean(axis=1) # Mean over models
            div = pdfs.sum(axis=1)[:,None]
            pdfs /= div            
            
            cdfs = np.clip(np.cumsum(pdfs, axis=1), 0.0, 1.0)
            cdfs = np.round(cdfs, decimals=8) # Float err causing issues with 1.0 - x
            sfs = 1.0 - cdfs
 
            p1 = sfs[:,t] 
            p1 /= p1.sum() or 1.0
            #p = np.array(p1)
            p1[k1] = 0.0
            
            p2 = sfs[:,t2] 
            p2 /= p2.sum() or 1.0
            p2[k1] = 0.0
            p2[k2] = 0.0

            p_values1[i] = max(0.0, p1.sum()) # Single # can be < 0.0 due to float error
            p_values2[i] = max(0.0, p2.sum()) # Dual            
            
            if fit:
                singularity[i] = _ensemble_survival(rv, prot_scores[:,k1], distrib_points)[int(single_thresh*fit_distrib_points)]
                duality[i] = _ensemble_survival(rv, prot_scores[:,(k1, k2)].sum(axis=1), distrib_points)[int(single_thresh*fit_distrib_points)]
            else:
                singularity[i] = np.count_nonzero(prot_scores[:,k1] > single_thresh)/m
                duality[i] = np.count_nonzero(prot_scores[:,(k1, k2)].sum(axis=1) > single_thresh)/m
 
            if singularity[i] > 0.5 and mean_scores[k1] > single_thresh:
                prediction[i] = k1
 
            elif duality[i] > 0.5:
                prediction[i] = k1
                prediction2[i] = k2

            if i % 100 == 0:
                self.info(f'{i:,} {n:,}', end='\r')
 
        is_single = singularity > 0.1
        p_values = p_values2.copy()
        p_values[is_single] = p_values1[is_single]
        
        init_profiles = self.train_profiles
        completeness = np.count_nonzero(np.nan_to_num(init_profiles), axis=1)/float(init_profiles.shape[1])
        novelty = completeness * p_values
        
        self.set_array_data('pval', p_values)
        self.set_array_data('pval1', p_values1)
        self.set_array_data('pval2', p_values2)
        
        self.set_array_data('novelty', novelty)
        self.set_array_data('completeness', completeness)
        self.set_array_data('singularity', singularity)
        self.set_array_data('duality', duality)

        self.set_marker_data('prediction2', prediction2, klass_labels)
        self.set_marker_data('prediction', prediction, klass_labels)
        

    def make_profile_predictions(self, split_idx=None, max_missing=0.35):

        # Input data
        klass_labels = self.train_labels
        train_klasses = self.train_markers
        init_profiles = self.train_profiles
        n, m = init_profiles.shape
        
        recon_label = self.recon_profile_key
        if not self.have_profile_label(recon_label):
            self.warn(f'Cannot find inference/predictions {recon_label}')
            return
 
        # Separate exp conditions for inspection
        if split_idx:
            for label, start, end in split_idx:
                self.set_profile_data(f'init_{label}', init_profiles[:,start:end])
 
        # Output data, maximal, unfiltered
        msg = 'Loading inference scores'
        self.info(msg)
        recon_profiles = self.recon_profiles
        latent_profiles = self.latent_profiles
        pred_score_ensemble = self.class_ensemble_preds  
        
        # Recon profile is median
        recon_profiles = np.median(recon_profiles, axis=1)
        latent_profiles = np.mean(latent_profiles, axis=1)
        
        # Average class assignments
        mean_klass_scores = pred_score_ensemble.mean(axis=1)
 
        # best class, best score
        best_klass_idx = mean_klass_scores.argmax(axis=-1)
        best_klass_score = mean_klass_scores.max(axis=-1)
 
        # Confusion matrix, for sanity check 
        self._print_confusion_matrix(train_klasses, best_klass_idx)
 
        pred_klass_labels = self.get_marker_labels('predictions')
        
        if pred_klass_labels != klass_labels:
            self.info(f'Re-indexing class labels to match predictions')
 
            idx_map = {-1:-1} # Old to new
            for i, x in enumerate(klass_labels):
                idx_map[i] = pred_klass_labels.index(x)
 
            train_klasses = np.array([idx_map[i] for i in train_klasses])
            klass_labels = pred_klass_labels
            
            #self.set_marker_data(marker_label, train_klasses, klass_labels)
         
        self._print_confusion_matrix(train_klasses, best_klass_idx)
        
        # Where prediction doesn't match training basic multplicity
        msg = 'Getting mismatches'
        self.info(msg)
        mismatches = np.full(n, -1, int)
        different = (train_klasses != best_klass_idx) & (train_klasses >= 0)
        mismatches[different] = best_klass_idx[different]
        self.set_marker_data('mismatches', mismatches, klass_labels)
 
        # Basic multiplicity
        msg = 'Getting Multiplicity'
        self.info(msg)
        multiplicity = np.count_nonzero(mean_klass_scores > 0.3, axis=-1)
        multiplicity[best_klass_score > 0.8] = 1
        self.set_marker_data('multiplicity', multiplicity, ['m%d' % x for x in range(multiplicity.max())])
 
        # Missing content of data
        msg = 'Getting zero counts'
        self.info(msg)
        zeros_count = np.count_nonzero(np.nan_to_num(init_profiles) == 0, axis=1)
        zeros_count = np.clip(zeros_count, 0, 20)
        self.set_marker_data('nzeros', zeros_count, ['z%d' % x for x in range(zeros_count.max()+1)])
 
        msg = 'Getting zero-filled profiles'
        # Fill-in only missing values
        is_missing = np.nan_to_num(init_profiles) <= 0.0
        zfill_profiles = np.array(init_profiles)
        zfill_profiles[is_missing] = recon_profiles[is_missing]
        self.info(msg)
        
        # Missing reconstruction
        for maxnan in (10,20,30,40,50):
            invalid = np.count_nonzero(np.isnan(init_profiles), axis=-1) > maxnan
            new_profiles =    np.array(zfill_profiles)
            new_profiles[invalid,:] = 0.0
            self.set_profile_data(f'zfill_maxnan{maxnan}', new_profiles)
 
        # Profiles with missing values filled in
        # Set a limit to how many values can be reconstructed
        # this is a bit conservative
        invalid = np.count_nonzero(np.isnan(init_profiles), axis=-1) > int(max_missing*m) # E.g. one third
        new_profiles =    np.array(zfill_profiles)
        new_profiles[invalid,:] = 0.0
        self.set_profile_data(self.zfill_profile_key, new_profiles)
 
        if split_idx:
            for label, start, end in split_idx:
                self.set_profile_data(f'zfill_{label}', new_profiles[:,start:end])
 
        # Reconstruction, conservative
        msg = 'Getting conservative reconstruction'
        self.info(msg)
        new_profiles =    np.array(recon_profiles)
        new_profiles[invalid,:] = 0.0
        self.set_profile_data(self.recon_profile_key, new_profiles)
 
        # Latent map, conservative
        msg = 'Getting conservative latent map'
        self.info(msg)
        latent_profiles[invalid,:] = 0.0
        self.set_profile_data(self.latent_profile_key, latent_profiles)
    
    
    def reconstruction_overview(self, tag, max_missing=0.35):    

        init_profiles = self.get_profile_data('init')
        init_profiles = np.nan_to_num(init_profiles)
        
        recon_profiles = self.get_profile_data('recon')
        
        n, m = recon_profiles.shape
        
        num_nz = np.count_nonzero(init_profiles, axis=-1)
        pc_zeros = 100.0 * (m - num_nz).sum()/ float(n*m)
        mean_nz = np.mean(num_nz)
        q25_nz, q75_nz = np.quantile(num_nz, [0.25, 0.75])
        
        valid = num_nz >= int((1.0-max_missing)*m)
                
        init_profiles    = init_profiles[valid]
        recon_profiles = recon_profiles[valid]

        n1 = len(init_profiles)

        num_valid_nz = np.count_nonzero(init_profiles, axis=-1)
        pc_recon_zeros = 100.0 * (m - num_valid_nz).sum()/ float(n1*m)
        mean_valid_nz = np.mean(num_valid_nz)
        q25_valid_nz, q75_valid_nz = np.quantile(num_valid_nz, [0.25, 0.75])
        
        #is_missing = np.nan_to_num(init_profiles) <= 0.0
     
        errs = np.abs(recon_profiles - init_profiles)
        q25, q75 = np.quantile(errs, [0.25, 0.75])
        mean_err = np.mean(errs,)
        
        sufficiency = 100.0 * n1 / n
        
        self.info('ALL DATA')
        self.info(f'{tag} Rows:{n:,} Cols:{m} Zeros:{pc_zeros:.1f}% Mean Row Non-zero:{mean_nz:.1f} IQR:{q25_nz:.1f}-{q75_nz:.1f} Rows Non-zero>65%:{sufficiency:.2f}')
        
        self.info('SUFFICIENT DATA')
        self.info(f'{tag} Rows:{n1:,} Cols:{m} Zeros:{pc_recon_zeros:.1f}% Mean Row Non-zero:{mean_valid_nz:.1f} IQR:{q25_valid_nz:.1f}-{q75_valid_nz:.1f} Abs. Recon. error {mean_err:.2f} IQR:{q25:.2f}-{q75:.2f}')
        
    
    def save_pruned_table(self, tsv_path):
        
        marker_idx = self.raw_markers        
        klass_idx = self.train_markers
        klass_lbl = self.train_labels + ['UNKNOWN']
        pred_data = self.class_ensemble_preds
        
        n, m, k = pred_data.shape
            
        pvals = self.get_array_data('pval')
        scores = pred_data.mean(axis=1)     
        sort_idx = scores.argsort(axis=1)
        first    = sort_idx[:,-1]
        second = sort_idx[:,-2]
        third    = sort_idx[:,-3]        
        
        pids = self.proteins
        title_dict = self.get_prot_titles(pids) 
             
        table_data = []
        
        for i in range(n):
            mk = marker_idx[i]
            tk = klass_idx[i]
           
            if mk == -1:
                    continue
           
            if mk == tk:
                    continue
        
            a = first[i]
            b = second[i]
            c = third[i]             

            table_data.append([klass_lbl[mk], klass_lbl[a], scores[i,a], klass_lbl[b], scores[i,b], klass_lbl[c], scores[i,c], pvals[i], pids[i], title_dict[pids[i]]])
        
        table_data.sort()
        table_heads = ['p-value', 'Pruned Marker Label', 'Pred Class 1', 'Score 1', 'Pred Class 2', 'Score 2', 'Pred Class 3', 'Score 3' 'Protein ID', 'Description']
        
        with open(tsv_path, 'w') as out_file_obj:
            out_file_obj.write('\t'.join(table_heads) + '\n')
     
            for mk, k1, s1, k2, s2, k3, s3, p, pid, desc in    table_data:
                if s3 < 0.05:
                    s3 = ''
                    k3 = ''
                    
                else:
                    s3 = f'{100.0*s3:.2f}'

                if s2 < 0.05:
                    s2 = ''
                    k2 = ''
                    
                else:
                    s2 = f'{100.0*s2:.2f}'
                        
                line = f'{p:.7f}\t{mk}\t{k1}\t{100.0*s1:.2f}\t{k2}\t{s2}\t{k3}\t{s3}\t{pid}\t{desc}\n'
                out_file_obj.write(line)
                         
        self.info(f'Write {len(table_data):,} lines to {tsv_path}')
     
    
    def plot_l2_loss_distrib(self, klass_label=None, plim=1e-2):

        profiles = np.nan_to_num(self.train_profiles)[:,:10]
        
        if klass_label is None:
            klass_label = self.raw_markers_key
        
        pred_data = self.class_ensemble_preds
        klass_idx = self.get_marker_data(klass_label)
        klass_lbl = self.get_marker_labels(klass_label)
        partition_idx = self.train_groups
        
        npt = int(max(partition_idx))
 
        pvals = self.get_array_data('pval')

        valid = ((klass_idx >= 0) & (klass_idx < pred_data.shape[2])).nonzero()[0]
        klass_idx = klass_idx[valid]
        pvals = pvals[valid]
        pred_data = pred_data[valid].mean(axis=1) # Mean over models/samples; protes x klasses
        pred_data /= pred_data.sum(axis=1)[:,None]
        partition_idx = partition_idx[valid]
        profiles = profiles[valid]
        pids = self.proteins
        pids = [pids[i] for i in valid]
 
        sq_diffs = []
        part_pvals = []
        idx = np.arange(len(pred_data))
 
        for i in range(npt): # Partitions
            selection = (partition_idx == i + 1)
 
            test_preds = pred_data[selection]
 
            idx2 = idx[selection]
 
            ideal_data = np.zeros(test_preds.shape)
 
            rows = np.arange(len(ideal_data))
 
            ideal_data[rows,klass_idx[selection]] = 1.0
 
            sq_diff = (ideal_data - test_preds) ** 2
 
            part_p = pvals[selection]
 
            sq_diffs.append(sq_diff)
            part_pvals.append(part_p)
 
            part_l2 = sq_diff.sum(axis=1)
 
            query = ((part_l2 < 0.03) & (part_p > 0.5)).nonzero()[0]

            xvals = range(ideal_data.shape[1])
 
            for j in query:
                i2 = idx2[j]
                fig, (ax1, ax2) = plt.subplots(1,2)
                ax1.plot(profiles[i2], alpha=1.0)
                title = f'{pids[i2]} : {klass_lbl[klass_idx[i2]]} : p={part_p[j]:.3f}'
                ax1.set_title(title)
                ax2.scatter(xvals, ideal_data[j], alpha=0.25)
                ax2.scatter(xvals, test_preds[j], alpha=1.0, s=3)
                title = f'L2={part_l2[j]:.5f}'
                ax2.set_title(title)
                plt.show()
 
        sq_diff = np.concatenate(sq_diffs)
        part_pvals = np.concatenate(part_pvals)
 
        good = part_pvals < plim
        bad = part_pvals >= plim
 
        l2s = sq_diff.sum(axis=1)
        l2s_good = sq_diff[good].sum(axis=1)
        l2s_bad = sq_diff[bad].sum(axis=1)
 
        fig, (ax1, ax2) = plt.subplots(2)
 
        hist, edges = np.histogram(l2s_good, range=(0.0, 1.0), bins=200)
        hist = hist.astype(float)
        hist /= hist.sum()
        ax1.plot(edges[:-1], hist, label=f'p < {plim}', alpha=0.5)
 
        hist, edges = np.histogram(l2s_bad, range=(0.0, 1.0), bins=200)
        hist = hist.astype(float)
        hist /= hist.sum()
        ax1.plot(edges[:-1], hist, label=f'p >= {plim}', alpha=0.5)
        ax1.legend()
 
        ax2.set_ylabel('Slingle-class p-value')
        ax2.set_xlabel('L2 loss')
        ax2.scatter(l2s, part_pvals, color='#0080FF', alpha=0.5, s=3)
 
        plt.show()
    
    
    def plot_overview(self, p1=1e-4, p2=0.1, min_nonzero=0.65, s_thresh=0.8):
    
        # Pie chart of % markers, conf, non-conf, missing/unknown
        # Singe prediction p-value ranges
        # p_values, novelty, completeness, p_values1, p_values2, singularity, duality
 
        title = self.source_tag
        
        klass_idx = self.train_markers
        klass_lbl = self.train_labels
        pred_data = self.class_ensemble_preds
        
        n, m, k = pred_data.shape

        pvals = self.get_array_data('pvals')
        pvals1 = self.get_array_data('pvals1')
        completeness = self.get_array_data('completeness')
        
        incomplete = completeness < min_nonzero
        complete = completeness >= min_nonzero
        trained = klass_idx >= 0
        untrained = klass_idx < 0
        predicted = complete & untrained

        scores = pred_data.mean(axis=1)
        medians = np.median(pred_data, axis=1)
        sort_idx = scores.argsort(axis=1)
        idx = np.arange(n, dtype=int)
        first    = sort_idx[:,-1]
        second = sort_idx[:,-2]
 
        single = (medians[idx,first] > s_thresh) & (scores[idx,first] > s_thresh)
        double = ((scores[idx,first] + scores[idx,second]) > s_thresh) & ~single
 
        single &= predicted
        double &= predicted
        other = predicted & ~(single | double)
 
        p_gud = (pvals < p1) & predicted
        p_med = (pvals >= p1) & (pvals <= p2) & predicted
        p_bad = (pvals > p2) & predicted
 
        n_inc = np.count_nonzero(incomplete)
        n_gud = np.count_nonzero(p_gud)
        n_med = np.count_nonzero(p_med)
        n_bad = np.count_nonzero(p_bad)
        n_trn = np.count_nonzero(trained)
 
        vals = [n_trn, n_gud, n_med, n_bad, n_inc]
        n0 = np.count_nonzero(complete)
 
        n_classd = n_trn + n_gud
 
        counts =    [n, n_inc, n-n_inc, n_classd, n_trn, n_gud, n_med, n_bad]
        fracs = [n/n, n_inc/n, (n-n_inc)/n, n_classd/n0, n_trn/n0, n_gud/n0, n_med/n0, n_bad/n0]
        pcs = [100.0 * x for x in fracs]
        slabels = ['Total','Data deficient','Data sufficient','Classified','Training','Good','Mediocre','Uncertain']
 
        self.info(f'{title}', end=' ')
 
        for slabel, count, pc in zip(slabels, counts, pcs):
                self.info(f'{slabel}:{count:,} ({pc:.1f}%)', end=' ')
 
        self.info('')
 
        fig, ax = plt.subplots()
 
        ax.set_title(title + f' n={n:,}')
 
        labels = ['Trained', f'p < {p1}', f'p [{p1}, {p2}]', f'p > {p2}','Data deficient']
        labels = ['Trained','Good','Mediocre','Uncertain','Data poor']
        colors = ['#400080','#0080FF','#B0B000','#E00000','#808080']
 
        for i, l in enumerate(labels):
             pc = 100.0 * vals[i] / float(n)
             labels[i] = f"{l}\nn={vals[i]:,}\n{pc:.2f}%"
 
        ax.pie(vals, wedgeprops=dict(width=0.4, edgecolor='w'), textprops=dict(color="w"),
              labeldistance=0.75, radius=1.0, labels=labels, colors=colors)
 
        n_single1 = np.count_nonzero(single & p_gud)
        n_double1 = np.count_nonzero(double & p_gud)
        n_other1 = np.count_nonzero(other & p_gud)
 
        n1 = float(n_single1 + n_double1 + n_other1)
        pc_single1 = 100.0 * n_single1/n1
        pc_double1 = 100.0 * n_double1/n1
        pc_other1    = 100.0 * n_other1/n1
 
        n_single2 = np.count_nonzero(single & p_med)
        n_double2 = np.count_nonzero(double & p_med)
        n_other2    = np.count_nonzero(other & p_med)
 
        n_single3 = np.count_nonzero(single & p_bad)
        n_double3 = np.count_nonzero(double & p_bad)
        n_other3    = np.count_nonzero(other & p_bad)
 
        self.info(f'{title} Single:{n_single1:,} ({pc_single1:.1f}%) Double:{n_double1:,} ({pc_double1:.1f}%) Other:{n_other1:,} ({pc_other1:.1f}%)')
 
        c1 = '#00FF00'
        c2 = '#00B000'
        c3 = '#008000'
 
        vals2 = [n_trn, n_single1, n_double1, n_other1, n_single2, n_double2, n_other2, n_single3, n_double3, n_other3, n_inc]

        colors2 = ['#FFFFFF', c1, c2, c3, c1, c2, c3, c1, c2, c3,'#FFFFFF']
 
        ax.pie(vals2, wedgeprops=dict(width=0.2, edgecolor='w'), rotatelabels=True, radius=0.6, colors=colors2)
 
        m1 = ax.scatter([], [], label='Single', marker='s', color=c1)
        m2 = ax.scatter([], [], label='Double', marker='s', color=c2)
        m3 = ax.scatter([], [], label='Other', marker='s', color=c3)
 
        ax.legend([m1, m2, m3], ['Single', 'Double', 'Other'], fontsize=10, loc=10)
        ax.set_axis_off()
 
        plt.show()
 
        fig, ax = plt.subplots()
 
        ax.hist(pvals1, bins=200)
 
        plt.show()
 

    def plot_dual_localisation_overview(self, p1=1e-4, s_thresh=0.8, min_nonzero=0.65, fontsize=8, save_path=None, tsv_path=None):
    
        #from scipy.cluster import hierarchy
        #from scipy.spatial import distance
        
        tag = self.source_tag
        
        cmap = LinearSegmentedColormap.from_list(name='CMAP01', colors=['#FFFFFF', '#4080FF'], N=25)
        cmap.set_bad('#DDDDCD', 1.0)

        marker_idx = self.raw_markers
        klass_idx = self.train_markers
        klass_lbl = self.train_labels + ['UNKNOWN']
        pred_data = self.class_ensemble_preds
        
        n, m, k = pred_data.shape
 
        pvals = self.get_array_data('pvals')
        completeness = self.get_array_data('completeness')
        complete = completeness >= min_nonzero
        untrained = klass_idx < 0
        predicted = complete & untrained
 
        scores = pred_data.mean(axis=1)
        medians = np.median(pred_data, axis=1)
        sort_idx = scores.argsort(axis=1)
        idx = np.arange(n, dtype=int)
        first    = sort_idx[:,-1]
        second = sort_idx[:,-2]
        third    = sort_idx[:,-3]
 
        single = (medians[idx,first] > s_thresh) & (scores[idx,first] > s_thresh)
        double = ((scores[idx,first] + scores[idx,second]) > s_thresh) & ~single
 
        matrix = np.zeros((k,k), float)
 
        pids = self.proteins
        title_dict = self.get_prot_titles(pids)
 
        table_out = []
 
        for i in range(n):
            if not predicted[i]:
                    continue
 
            if not double[i]:
                    continue
 
            if pvals[i] > p1:
                    continue
 
            a = first[i]
            b = second[i]
            c = third[i]
 
            matrix[a,b] += 1
            matrix[b,a] += 1
 
            t = marker_idx[i]
 
            if t >= 0:
                    tklass = klass_lbl[t]
            else:
                    tklass = ''
            
            row = [klass_lbl[a], klass_lbl[b], pvals[i], klass_lbl[c],
                   scores[i,a], scores[i,b], scores[i,c],
                   tklass, pids[i], title_dict[pids[i]]]
            table_out.append(row)
 
        if tsv_path:
            table_out.sort()
            table_heads = ['p-value', 'Primary Class', 'Score 1', 'Secondary Class', 'Score 2',
                           'Class 3 (score >5%)', 'Score 3', 'Unfiltered Marker', 'Protein ID', 'Description']
 
            with open(tsv_path, 'w') as out_file_obj:
                out_file_obj.write('\t'.join(table_heads) + '\n')
 
                for k1, k2, p, k3, s1, s2, s3, tklass, pid, desc in    table_out:
                    if s3 < 0.05:
                        s3 = ''
                        k3 = ''
 
                    else:
                        s3 = f'{100.0*s3:.2f}'
 
                    line = f'{p:.7f}\t{k1}\t{100.0*s1:.2f}\t{k2}\t{100.0*s2:.2f}\t{k3}\t{s3}\t{tklass}\t{pid}\t{desc}\n'
                    out_file_obj.write(line)
 
            self.info(f'Write {len(table_out):,} lines to {tsv_path}')
 
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0.17, bottom=0.17, right=0.93, top=0.93, wspace=0.1, hspace=0.1)
        fig.set_size_inches(7.0, 7.0)
 
        m = np.log10(matrix.max())
 
        sort_labels = klass_lbl
        
        #corr_mat = distance.pdist(matrix, metric='correlation')
        #linkage = hierarchy.ward(corr_mat)
        #order = hierarchy.leaves_list(linkage)
        #sort_labels = [klass_lbl[j] for j in order]
        #matrix = matrix[order[::-1]][:,order[::-1]]
 
        for a in range(k):
            for b in range(k):
                if matrix[a,b] > 0:
                    v = np.log10(matrix[a,b])
                    color = '#FFFFFF' if (v/m) > 0.55 else '#000000'
                    ax.text(a, b, int(matrix[a,b]), color=color, va='center', ha='center', fontsize=9)
                    matrix[a,b] = v
 
                else:
                    matrix[a,b] = np.nan

        img = ax.imshow(matrix.T, cmap=cmap, origin='lower')
 
        xtick_pos = np.arange(k)
        ytick_pos = np.arange(k)
 
        ax.set_title(f'Dual location predictions : {tag}')
 
        ax.set_xticks(xtick_pos)
        ax.set_yticks(ytick_pos)
 
        ax.set_xticklabels(sort_labels, rotation=90, fontsize=fontsize)
        ax.set_yticklabels(sort_labels, fontsize=fontsize)
 
        ax.xaxis.set_label_position("top")
        ax.yaxis.set_label_position("right")
 
        if save_path:
            plt.savefig(save_path, dpi=400)
            self.info(f'Saved {save_path}')
 
        else:
            plt.show()
 
 
    def plot_pr_curve(self):
                
        klass_labels = self.train_labels
        true_klasses = self.train_markers
        
        # 0 index is not used in trainaing, > 0 reflects leave-chunk-out group, e.g. 1 - 10
        partition_idx = self.train_groups
        
        n = int(max(partition_idx))
        
        scores = self.class_ensemble_preds    
        
        predictions = scores[:,:,:len(klass_labels)]
        
        prots, m, n_klasses = predictions.shape
        
        dm = int(m//n) # Models per partition
        
        self.info(f'Using {m:,} predictions, from {n} partitions, of {n_klasses} classes in {prots:,} proteins')
        
        fig, ax = plt.subplots()
        
        ax.set_xlabel('Class Recall')
        ax.set_ylabel('Class Precision')
        
        small = np.logspace(-82.0, -2.0, 200)
        almost_one = 1.0 - small[::-1]
        sort_scores = np.sort(np.concatenate([small, np.linspace(0.01, 0.99, 1000), almost_one]))
        sort_scores = np.linspace(0.01, 0.99, 99)
        n_points = len(sort_scores)
        mid_point = np.searchsorted(sort_scores, 0.5001)
        
        klass_f1s = []
        
        for k in range(n_klasses):
            
            klass = klass_labels[k]
            color = COLOR_DICT[klass]
            rgb = ColorConverter.to_rgb(color)
            h, l, s = colorsys.rgb_to_hls(*rgb)
            color = colorsys.hls_to_rgb(h, 0.8 * l, s)
            
            f1s = []
            
            recalls = np.zeros(n_points+1, float)
            rcounts = np.zeros(n_points+1, float)
            pcounts = np.zeros(n_points+1, float)
            precisions = np.zeros(n_points+1, float)

            for i in range(n): # Partitions
                selection = (partition_idx == i + 1) & (true_klasses >=0 )
                tru_klass = true_klasses[selection]
                tru = tru_klass == k
                num_t = sum(tru.astype(int))

                if not num_t:
                    continue
 
                p = i * dm
                q = p + dm
 
                for i2 in range(p, q):
                    test_scores = predictions[selection,i2,:]
                    #test_scores = predictions[selection,p:q].mean(axis=1)
                    test_scores /= test_scores.sum(axis=1)[:,None]
                    pred_scores = test_scores[:,k]
                    true_scores = pred_scores[tru]

                    for s, tval in enumerate(sort_scores):
                       num_p =    np.count_nonzero((pred_scores >= tval).astype(int))
                       num_tp = np.count_nonzero((true_scores >= tval).astype(int))
 
                       if num_t:
                           recall = num_tp / num_t
                           recalls[s] += num_tp / num_t
                           rcounts[s] += 1.0
                       else:
                           recall = 0.0
                               
                       if num_p:
                          precision = num_tp / num_p
                          precisions[s] += precision
                          pcounts[s] += 1.0
                          
                          if num_tp and s == mid_point:
                              f1 = (2 * precision * recall) / (precision +recall )
                              f1s.append(f1)
                                             
            klass_f1s.append(np.mean(f1s))
            
            # Macro = mean
            valid = rcounts > 0
            recalls[valid] /= rcounts[valid]
 
            valid = pcounts > 0
            precisions[valid] /= pcounts[valid]
            precisions[~valid] = 1.0
 
            recall_05 = recalls[mid_point]
            precision_05 = precisions[mid_point]
            
            f1 = (2 * precision_05 * recall_05) / (precision_05 + recall_05)
            
            marker_count = np.count_nonzero(true_klasses == k)
            
            label = f'{klass} F$_1$={f1:.3f} k={marker_count:,}'
            ax.plot(recalls, precisions, label=label, alpha=0.4, linewidth=2, color=color)
            ax.scatter([recall_05], [precision_05], s=40, color=color, marker='*', zorder=9)
            ax.scatter([recall_05], [precision_05], s=5, color='k', marker='*', zorder=90)

        ax.set_xlim(0.65, 1.02)
        ax.set_ylim(0.65, 1.02)
        
        ax.legend(fontsize=8)
        
        macro_f1s = np.array(klass_f1s)
        
        title = f'Macro F$_1$ = {macro_f1s.mean():.3f} \u00B1 {macro_f1s.std():.3f} ($n=${len(macro_f1s):,})'
        
        ax.set_title(title)
        
        plt.show()
            

    def plot_contingency_table(self, marker_label, marker_title='Training Class',
                               save_path=None, fontsize=8):

        cmap = LinearSegmentedColormap.from_list(name='CMAP01', colors=['#FFFFFF', '#4080FF'], N=25)
        cmap.set_bad('#DDDDCD', 1.0)
        
        pred_labels = self.class_ensemble_preds
        
        marker_labels = self.get_marker_labels(marker_label)
        
        all_labels = set(pred_labels) & set(marker_labels) - set(['unknown','DUAL'])
        all_labels = sorted(all_labels) + ['undefined']
        u = len(all_labels) - 1
        
        idx = {x:i for i, x in enumerate(all_labels)}
        
        markers = self.get_marker_data(marker_label)        
        predictions = self.class_ensemble_preds.mean(axis=1).argmax(axis=-1)
        
        # Map separate to common idx ; old to new
        map_pred_idx    = {i:idx[x] for i, x in enumerate(pred_labels)}
        map_marker_idx = {i:idx[x] for i, x in enumerate(marker_labels)}
        map_pred_idx[1+len(pred_labels)] = u
        map_marker_idx[1+len(marker_labels)] = u
        map_pred_idx[len(pred_labels)] = u
        map_marker_idx[len(marker_labels)] = u
        

        valid = (markers >= 0) | (predictions >= 0)
        
        markers = markers[valid]
        predictions = predictions[valid]
        
        markers[markers < 0] = u
        predictions[predictions < 0] = u

        n = u + 1
        matrix = np.zeros((n,n), float)
        
        # Paired indices
        for j, k in zip(markers, predictions):
            a = map_marker_idx[j]
            b = map_pred_idx[k]
            matrix[a,b] += 1
        
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0.17, bottom=0.17, right=0.93, top=0.93, wspace=0.1, hspace=0.1)        
        fig.set_size_inches(7.0, 7.0)
        
        m = np.log10(matrix.max()) 
        for j in range(n):
             a = map_marker_idx[j]
            
             for k in range(n):
                 b = map_pred_idx[k]
                
                 if matrix[a,b] > 0:
                     v = np.log10(matrix[a,b])
                     color = '#FFFFFF' if (v/m) > 0.55 else '#000000'
                     t = ax.text(a, b, int(matrix[a,b]), color=color, va='center', ha='center', fontsize=9)
                     matrix[a,b] = v
                 else:
                     matrix[a,b] = np.nan
        
        totl = np.nansum(matrix)
        same = np.nansum(np.diag(matrix))
        diff = totl - same
        
        identity = 100.0 * diff/float(totl)
        
        img = ax.imshow(matrix.T, cmap=cmap, origin='lower')
        
        xtick_pos = np.arange(n)
        ytick_pos = np.arange(n)
        
        ax.set_title(f'{identity:.1}% same', loc='right')
        
        ax.set_xticks(xtick_pos)
        ax.set_yticks(ytick_pos)
        
        ax.set_xticklabels(all_labels, rotation=90, fontsize=fontsize)
        ax.set_yticklabels(all_labels, fontsize=fontsize)
        
        ax.xaxis.set_label_position("top")
        ax.yaxis.set_label_position("right")
        
        ax.set_xlabel(marker_title, fontsize=14, va="bottom")
        ax.set_ylabel('Predicted Class', fontsize=14, rotation=270, va="bottom")
        
        if save_path:
            plt.savefig(save_path, dpi=400)
            self.info(f'Saved {save_path}')
        
        else:        
            plt.show()
         
    
    def plot_mixed_ave_profiles(self, class1, class2, replica=0, min_nonzero=0.8, score_min=0.8, save_paths=None):
 
        quants = [0.25, 0.5, 0.75]
 
        prof_data = self.train_profiles
        valid = self.get_valid_mask(min_nonzero)
        labels = self.train_labels
 
        replica_widths = self.replica_cols

        replica_starts = np.cumsum(replica_widths)
        replica_ends = replica_starts + replica_widths
        replica_data = prof_data[valid,replica_starts[replica]:replica_ends[replica]]
 
        scores = self.class_ensemble_preds[valid] # prot, replica, klass
        scores = scores.mean(axis=1)
        best_klass = scores.argmax(axis=1)
        best_score = scores.max(axis=1)
 
        x_vals = np.arange(1, replica_widths[replica]+1)

        fig, axarr = plt.subplots(3, 1)
        fig.set_size_inches(8,8)
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.1, hspace=0.3)
        pairs = [(class1, None), (class1, class2), (class2, None)]
 
        for row, (klass1, klass2) in enumerate(pairs):
            ax = axarr[row]
 
            if klass2:
                k1 = labels.index(klass1)
                k2 = labels.index(klass2)
 
                scores1 = scores[:,k1]
                scores2 = scores[:,k2]
                mix_scores = scores1 + scores2
                selection = (mix_scores > score_min) & (scores1 > 0.25) & (scores2 > 0.25)
 
                rgb1 = np.array(ColorConverter.to_rgb(COLOR_DICT[klass1]))
                rgb2 = np.array(ColorConverter.to_rgb(COLOR_DICT[klass2]))
                color = tuple((rgb1+rgb2)/2.0)
                label = f'{klass1} + {klass2}'
 
            else:
                k1 = labels.index(klass1)
                selection = (best_klass == k1) & (best_score > score_min)
 
                color = COLOR_DICT[klass1]
                label = klass1
 
            klass_data = replica_data[selection]
            lower, median, upper = np.nanquantile(klass_data, quants, axis=0)

            ax.fill_between(x_vals, lower, upper, color=color, alpha=0.3, label='IQR')

            ax.plot(x_vals, median, color=color, linewidth=2, label=f'Median')
            ax.set_title(f'Average Profile : {label} (n={len(klass_data):,})', fontsize=9)
 
            if row == 2:
                ax.set_xlabel('MS Fraction')
 
            ax.set_ylabel('Rel. abundance')
            ax.legend(fontsize=9, framealpha=0.2)
            ax.tick_params(axis='both', which='both', labelsize=8)
 
 
        if save_paths:
            save_path = save_paths.format(class1, class2)
            plt.savefig(save_path, dpi=200)
            self.info(f'Saved {save_path}')
            plt.close()
 
        else:
            plt.show()
    

    def plot_ave_profiles(self, marker_label, replica=0, save_paths='png/ave_profile_{}.png', min_nonzero=0.8):
        
        self._check_marker_label(marker_label)

        prof_data = self.train_profiles
 
        #self.info(f'Data size for "{prof_label}": {prof_data.shape}')
 
        valid = self.get_valid_mask(min_nonzero)
 
        klasses = self.get_marker_data(marker_label)[valid]
        labels = self.get_marker_labels(marker_label)
 
        replica_widths = self.replica_cols
 
        replica_starts = np.cumsum(replica_widths)
        replica_ends = replica_starts + replica_widths
        replica_data = prof_data[valid,replica_starts[replica]:replica_ends[replica]]
 
        x_vals = np.arange(1, replica_widths[replica]+1)
 
        #plt.style.use('dark_background')
 
        npanels = len(labels)
        ncols = 3
        nrows = int(math.ceil(npanels/float(ncols)))
 
        fig, axarr = plt.subplots(nrows, ncols)
        fig.set_size_inches(10.0, 8.0)
        fig.subplots_adjust(left=0.08, bottom=0.08, right=0.95, top=0.95, wspace=0.15, hspace=0.3)
 
        for k, label in enumerate(labels):
            row = k % nrows
            col = k // nrows
            ax = axarr[row, col]
 
            idx = (klasses == k).nonzero()[0]

            if not len(idx):
                continue
 
            klass_data = replica_data[idx]
            lower, median, upper = np.nanquantile(klass_data, [0.25, 0.5, 0.75], axis=0)

            color = COLOR_DICT[label]
            ax.fill_between(x_vals, lower, upper, color=color, alpha=0.3, label='IQR')

            ax.plot(x_vals, median, color=color, linewidth=2, label=f'Median')
            ax.set_title(f'Average Profile : {label} (n={len(klass_data):,})', fontsize=9, color=color)
 
            if (k == npanels -1 ) or (row == nrows-1):
                ax.set_xlabel('MS Fraction')
 
            if col == 0:
                ax.set_ylabel('Rel. abundance')
 
 
            if (row < nrows-1) and (k < len(labels)-1):
                ax.set_xticklabels([])
 
            ax.legend(fontsize=7, framealpha=0.2)
            ax.tick_params(axis='both', which='both', labelsize=8)
 
            if save_paths:
                save_path = save_paths.format(label)
                plt.savefig(save_path, dpi=200)
                self.info(f'Saved {save_path}')
                plt.close()
 
        k += 1
        while k < nrows * ncols:
            ax = axarr[k % nrows, k // nrows]
            ax.set_axis_off()
            k += 1
 
        if not save_paths:
            plt.show()

    
    def plot_prediction_scatter(self, n_cols=2, protein_ids=None, save_paths='png/class_scores_{}_{:03d}.png'):

        def _make_score_plot(ax, title, score_data, labels, col):

            m, k = score_data.shape

            if col == 0:
                ax.set_ylabel('DNN score')

            if len(title) > 64:
                fontsize = 10 * (64.0/len(title))
            else:
                fontsize = 10

            ax.set_title(title, fontsize=fontsize)

            for i in range(k):
                x_vals = np.random.uniform(-0.4, 0.4, m) + i
                y_vals = score_data[:,i]

                ax.scatter(x_vals, y_vals, alpha=0.5, s=3, color=COLOR_DICT[labels[i]], label=labels[i])
                ax.scatter([i], [np.mean(y_vals)], color='#000000', marker='*')
 
                ax.text(i-0.1, -0.07, labels[i], rotation=-90, color=COLOR_DICT[labels[i]],
                                fontsize=8,    va='top', ha='center').set_clip_on(False)
 
            ax.set_xticks(np.arange(k))
 
 
            #ax.set_xticklabels([x.strip() for x in labels], rotation=-45, fontsize=8, ha='left')
            ax.set_xticklabels([])
            ax.set_ylim(-0.05, 1.05)
 
        if protein_ids:
            pids = list(protein_ids)
            valid = np.array([True if pid in set(pids) else False for pid in self.proteins], bool)
            n_rows = int(math.ceil(len(pids)//n_cols))
 
        else:
            valid = np.count_nonzero(np.isnan(self.train_profiles), axis=-1) <= 28
            pids = [pid for i, pid in enumerate(self.proteins) if valid[i]]
            n_rows = max(5, int(math.ceil(len(pids)//n_cols)))
 
        pklasses = self.class_ensemble_preds[valid,::10]
        organelle_labels = self.train_labels    + ['unknown']
    
        title_dict = self.get_prot_titles(pids)
 
        # Average class assignments
        pred_classes = pklasses.mean(axis=1)
 
        # best class, best score
        best_idx = pred_classes.argmax(axis=-1)
        best_score = pred_classes.max(axis=-1)
 
        n, reps, nk = pklasses.shape
 
        prediction_classes = defaultdict(list)
 
        for i in range(n):
            s = best_score[i]
            k = best_idx[i]
 
            if protein_ids:
                key = 'Selection'
 
            elif s > 0.8:
                key = f'main_{organelle_labels[k]}'
 
            else:
                multiplicity = np.count_nonzero(pred_classes[i] > 0.2)
 
                if multiplicity == 1:
                    if s > 0.5:
                        key = f'marginal_{organelle_labels[k]}'
                    else:
                        key = f'unclear_{organelle_labels[k]}'
 
                elif multiplicity == 2:
                    k2, k1 = np.argsort(pred_classes[i])[-2:]
                    dual_key = sorted([organelle_labels[k1], organelle_labels[k2]])
                    dual_key = '+'.join(dual_key)
                    key = f'dual_{dual_key}'

                else:
                    key = f'unknown'
 
            prediction_classes[key].append((s, i))

        step = n_rows * n_cols
        for key in sorted(prediction_classes):
            sort_idx = prediction_classes[key]
            sort_idx.sort(reverse=False)

            idx = [x[1] for x in sort_idx]
            n = len(idx)
 
            for i in range(0, n, step):
 
 
                fig, axarr = plt.subplots(n_rows, n_cols)
                fig.set_size_inches(5.0 * n_rows, 6.0 * n_cols)
 
                for j in range(step):
                    k = i + j
                    pid = pids[idx[k]]
                    col = j % n_cols
                    row = j // n_cols
 
                    if k < n:
                        title = f'{pid} : {title_dict[pid]}'
                        _make_score_plot(axarr[row, col], title, pklasses[idx[k]], organelle_labels, col)
                    else:
                        axarr[row, col].set_axis_off()

                fig.subplots_adjust(left=0.1, bottom=0.12, right=0.95, top=0.95, wspace=0.1, hspace=0.34)
 
                if save_paths:
                    save_path = save_paths.format(key, i)
                    plt.savefig(save_path, dpi=400)
                    self.info(f'Saved {save_path}')
 
                else:
                    plt.show()
 
                plt.close(fig)
 
    
    def plot_reconstruction(self, save_path=None):

        pids = self.proteins
 
        recon_profile_data = self.recon_profiles
        ref_profile_data = self.train_profiles
 
        z = np.count_nonzero(np.isnan(ref_profile_data), axis=1)
 
        ncols = 5
        nrows = 4

        valid1 = (z > 0) & (z <= 7)
        valid2 = (z > 7) & (z <= 14)
        valid3 = (z > 14) & (z <= 21)
        valid4 = (z > 21) & (z <= 28)
 
        valid1 = valid1.nonzero()[0]
        valid2 = valid2.nonzero()[0]
        valid3 = valid3.nonzero()[0]
        valid4 = valid4.nonzero()[0]

        np.random.shuffle(valid1)
        np.random.shuffle(valid2)
        np.random.shuffle(valid3)
        np.random.shuffle(valid4)
 
        valid = np.concatenate([valid1[:ncols], valid2[:ncols], valid3[:ncols], valid4[:ncols]])
 
        row_tags = ['1-7','8-14','15-21','22-28']
 
        recon_profile_data = recon_profile_data[valid]
        ref_profile_data = ref_profile_data[valid]
 
        recon_profile_data -= np.nanmin(recon_profile_data, axis=1)[:,None]
        recon_profile_data /= np.nanmean(recon_profile_data, axis=1)[:,None]
        recon_profile_data *= np.nanmean(ref_profile_data, axis=1)[:,None]
 
        #ref_profile_data /= np.nanmax(ref_profile_data, axis=1)[:,None]
 
        n, p = ref_profile_data.shape
 
        n_plots = ncols*nrows
 
        pids = pids[:n_plots]
        title_dict = self.get_prot_titles(pids)
 
        #plt.style.use('dark_background')
 
        fig1, axarr = plt.subplots(nrows, ncols)
 
        for i, pid in enumerate(pids):
            col = i % ncols
            row = i // ncols
            ax = axarr[row, col]
            ax.scatter(recon_profile_data[i], ref_profile_data[i], color='#2080FF', s=8, alpha=0.85)
            if row == nrows-1:
                ax.set_xlabel('Reconstructed') # Add recon
            if col == 0:
                ax.set_ylabel(f'Original ({row_tags[row]} missing)')
            ax.set_title(title_dict[pid], fontsize=9)

        fig2, axarr2 = plt.subplots(nrows, ncols)
 
        blank = np.array([float('nan')] * p)
 
        for i, pid in enumerate(pids[:ncols*nrows]):
            col = i % ncols
            row = i // ncols
            ax = axarr2[row, col]
 
            x_vals = np.arange(1, p+1)
 
            recon_profile = recon_profile_data[i]
 
            orig_profile = ref_profile_data[i]
            missing = np.nonzero(np.isnan(orig_profile))[0]
            ms = set(missing)
            extra = set()
            for i in missing:
                if i > 0 and (i-1 not in ms):
                    extra.add(i-1)
                if i+1 < p and (i+1 not in ms):
                    extra.add(i+1)
 
            extra = np.array(sorted(extra))
 
            fill_profile = blank.copy()
            fill_profile[extra] = orig_profile[extra]
            fill_profile[missing] = recon_profile[missing]
 
            ax.plot(x_vals, recon_profile, color='#000000', alpha=0.9, linewidth=1, linestyle='--', label='Recon')
            ax.plot(x_vals, orig_profile, color='#0080FF', alpha=0.5, linewidth=2, label='Original')
            ax.plot(x_vals, fill_profile, color='#BBBB00', alpha=0.5, linewidth=2, label='Added')
 
            if col == 0:
                ax.set_ylabel(f'Abundance ({row_tags[row]} missing)')
            if row == nrows-1:
                ax.set_xlabel('Fraction')
 
            title = title_dict[pid]
            if len(title) > 64:
                fontsize = 8 * (64.0/len(title))
            else:
                fontsize = 8
 
            ax.set_title(title, fontsize=fontsize)
            ax.legend(fontsize=8, framealpha=0.2)
 
        fig1.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.25)
        fig2.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.25)
 
        if save_path:
            sf1 = save_path.format('scatter')
            sf2 = save_path.format('profiles')
 
            fig1.savefig(sf1, dpi=200)
            fig2.savefig(sf2, dpi=200)
            self.info(f'Saved images to {sf1}, {sf2}')
 
        else:
            plt.show()
 
    
    def plot_dual_proj_2d(self, profile_label, proj_method='umap', title=None,
                           min_nonzero=0.8, plot_size=8.0, spot_size=5, alpha=0.25,
                           save_path='png/Dual_loc_{}.png'):
        
        if not title:
            title = f'{proj_method.upper()} 2D'
 
        self._check_profile_label(profile_label)
        profile_data = self.get_profile_data(profile_label)
        
        valid = np.count_nonzero(np.isnan( self.train_profiles), axis=-1) <= 28
        profile_data = np.nan_to_num(profile_data)
 
        score_array = self.class_ensemble_preds
        class_labels = self.train_labels
 
        profile_data = profile_data[valid]
        score_array = score_array[valid]
        score_array = score_array.mean(axis=1)
 
        c1 = np.array([1.0,0.0,0.0])
        c2 = np.array([0.0,0.0,1.0])
        c3 = np.array([0.8,0.8,0.0])
        cbg = np.array([0.8,0.8,0.8])
 
        k = len(class_labels)
        nrows = int(math.sqrt(k))
        ncols = int(math.ceil(k/float(nrows)))
 
        proj_2d = self.get_profile_proj_2d(profile_label, proj_method)
        x_vals, y_vals = proj_2d.T
        x_vals = x_vals[valid]
        y_vals = y_vals[valid]
        n = len(x_vals)

        #plt.style.use('dark_background')

        for i, klass1 in enumerate(class_labels):
            fig, axarr = plt.subplots(nrows, ncols)
            fig.suptitle(f'Dual localisation with {klass1} n={len(profile_data):,}', fontsize=12)
            fig.set_size_inches(24.0, 16.0)
 
            fig0, ax0 = plt.subplots()
            ax0.set_title(f'Dual localisation with {klass1} n={len(profile_data):,}', fontsize=12)
            colors0 = np.zeros((n, 3)) + cbg
            zorder0 = np.zeros(n)
            pred1 = score_array[:,i]
 
            hex_color = COLOR_DICT[klass1]
            rgb1 = np.array([int(hex_color[1:3], 16),
                                             int(hex_color[3:5], 16),
                                             int(hex_color[5:7], 16)], float) /255.0
 
            for j, klass2 in enumerate(class_labels):
                col = j // nrows
                row = j % nrows
                ax = axarr[row, col]
 
                pred2 = score_array[:,j]
 
                intens = pred1 + pred2
                nz = intens > 0
 
                frac = np.zeros(intens.shape)
                frac[nz] = pred2[nz] / intens[nz]
 
                if i == j:
                    idx = (pred1 >= 0.8).nonzero()[0]
                    ax0.scatter([], [], color=rgb1, label=f'{klass1} ({len(idx):,})', marker='*', alpha=1.0, s=2*spot_size)
 
                else:
                    hex_color = COLOR_DICT[klass2]
                    rgb2 = np.array([int(hex_color[1:3], 16),
                                                     int(hex_color[3:5], 16),
                                                     int(hex_color[5:7], 16)], float) /255.0
 
                    selection = (pred1 < 0.8) & (pred2 < 0.8) & (pred1 > 0.35) & (pred2 > 0.35)
                    idx = selection.nonzero()[0]
                    colors0[idx] = rgb2 * intens[idx,None] + (1.0-intens[idx,None]) * cbg
                    zorder0[idx] = intens[idx]
                    ax0.scatter([], [], color=rgb2, label=f'{klass2} ({len(idx):,})', alpha=0.75, s=spot_size)

                colors = []
                for a, f in enumerate(frac):
                    if f < 0.5:
                        f *= 2.0
                        rgb = (1.0 - f) * c1 + f * c3
 
                    else:
                        f -= 0.5
                        f *= 2.0
                        rgb = (1.0 - f) * c3 + f * c2
 
                    colors.append(rgb * intens[a] + (1.0-intens[a]) * cbg)
 
                colors = np.array(colors)
                colors[intens < 0.5] = cbg
                #colors[errors > p_lim] = cbg
 
                colors = np.clip(colors, 0.0, 1.0)
                zorder = np.argsort(colors.sum(axis=-1))
                colors = [tuple(x) for x in colors[zorder]]
 
                ax.scatter(x_vals[zorder], y_vals[zorder], alpha=0.5, s=spot_size, color=colors, edgecolors='none')
                ax.set_title(klass2)
 
                if i == j:
                    ax.plot([], color=c3, label=klass1)
 
                else:
                    ax.plot([], color=c1, label=klass1)
                    ax.plot([], color=c3, label='Both')
                    ax.plot([], color=c2, label=klass2)
 
                if col > 0:
                    ax.set_yticklabels([])
 
                if (row < (nrows-1)) and (j < len(class_labels)-1):
                    ax.set_xticklabels([])
 
                ax.legend(fontsize=9, frameon=False)
 
            j += 1
            while j < ncols * nrows:
                ax = axarr[j % nrows, j // nrows]
                ax.set_axis_off()
                j += 1
 
            selected = zorder0 < 0.7
            ax0.scatter(x_vals[selected], y_vals[selected], alpha=0.4, s=spot_size,
                                    color=cbg, edgecolors='none', marker='*', zorder=0)
 
            selected = pred1 >= 0.8
            idx = selected.nonzero()[0]
            colors0[idx] = rgb1 * intens[idx,None] + (1.0-intens[idx,None]) * cbg
            colors0 = np.clip(colors0, 0.0, 1.0)
            ax0.scatter(x_vals[selected], y_vals[selected], alpha=0.4, s=2*spot_size,
                                    color=colors0[selected], edgecolors='none', zorder=2, marker='*')
 
            selected = zorder0 > 0.7
            ax0.scatter(x_vals[selected], y_vals[selected], alpha=0.75, s=spot_size,
                                    color=colors0[selected], zorder=5)
            ax0.legend(fontsize=5.5, frameon=False)

            if save_path:
                file_path = save_path.format(klass1.replace('/','or'))
                fig.subplots_adjust(left=0.03, bottom=0.05, right=0.97, top=0.95, wspace=0.1, hspace=0.1)
                fig.savefig(file_path, dpi=400)
                self.info(f'Saved {file_path}')
                file_path = save_path.format('comb_' + klass1.replace('/','or'))
                fig0.subplots_adjust(left=0.1, bottom=0.05, right=0.98, top=0.92, wspace=0.1, hspace=0.1)
                fig0.savefig(file_path, dpi=800)
                self.info(f'Saved {file_path}')
            else:
                plt.show(fig)

            plt.close(fig)
            plt.close(fig0)

     
    def plot_triple_2d(self, *args, **kw):
        
        args = args + (['umap','tsne','pca'])
        
        self. _plot_proj_2d(*args, **kw)
        
        
    def plot_tsne_2d(self, *args, **kw):
 
        args = args + (['tsne'],)
        
        self. _plot_proj_2d(*args, **kw)
        
 
    def plot_umap_2d(self, *args, **kw):

        args = args + (['umap'],)
        
        self. _plot_proj_2d(*args, **kw)

    
    def plot_pca_2d(self, *args, **kw):

        args = args + (['pca'],)
        
        self. _plot_proj_2d(*args, **kw)
             
    
    def get_profile_proj_2d(self, label, method, min_nonzero=0.5, recalc=False, tsne_perplexity=30.0,
                            umap_neighbours=7, umap_mindist=0.1, metric='correlation'):
        
        from sklearn.decomposition import PCA, FastICA
        from sklearn.manifold import TSNE
        from umap import UMAP

        self._check_profile_label(label)
        
        if not recalc:
            save_data = self.get_2d_proj(method, label)
                
            if save_data is not None:
                return save_data                
        
        profile_data = self.get_profile_data(label)
        profile_data = np.nan_to_num(profile_data)
        profile_data_all = profile_data.copy()
        
        n, m = profile_data.shape
        
        # Choose most complete data
        valid = np.count_nonzero(profile_data, axis=1) >= int(min_nonzero * m)
        profile_data = profile_data[valid]
 
        stdev = np.std(profile_data, axis=1)
        valid_rows = stdev > 0.0 # False for nan too
        
        # Normalise
        profile_data[valid_rows,] /= stdev[valid_rows,None]
        profile_data /= profile_data.max()
             
        if method == 'umap':
            kernel = UMAP(n_components=2, n_neighbors=umap_neighbours, min_dist=umap_mindist, metric=metric, random_state=7)
             
        elif method == 'tsne':
            kernel = TSNE(n_components=2, perplexity=tsne_perplexity, init='pca', metric=metric)

        elif method == 'pca':
            kernel = PCA(n_components=2)

        elif method == 'ica':
            kernel = FastICA(n_components=2)
        
        proj_model = kernel.fit(profile_data)
        
        proj_2d = proj_model.transform(profile_data_all) #x_vals, y_vals = proj_2d.T

        return self.set_2d_proj(proj_2d, method, label)
        
                                
    def get_profile_umap(self, label, recalc=False):
        
        return self.get_profile_proj_2d(label, 'umap', recalc)

    
    def write_prediction_tsv(self, out_file_path, pred_label='class_pred_all', score_label='combined_scores',
                             marker_labels=('prediction','prediction2'), marker_heads=('pred_primary_loc','pred_secondary_loc'),
                             score_heads=('p-value', 'single_score', 'dual_score')):
        
        if not self.has_pred_class_key(pred_label):
            self.warn(f'Cannot write prediction TSV file prediction data "{pred_label}" not known')
            return

        if not self.have_profile_label(score_label):
            self.warn(f'Cannot write prediction TSV file profile data "{score_label}" not known')
            return
        
        for label in marker_labels:
            if not self.has_marker_key(label):
                self.warn(f'Cannot write prediction TSV file marker track "{pred_label}" not known')
                return
                
        klass_labels = self.get_pred_class_labels(pred_label) + ['UNKNOWN']
        
        pred_data = self.get_pred_class_data(pred_label)
        
        combined_scores = self.get_profile_data(score_label)
     
        n_prots, n_models, n_klasses = pred_data.shape
        
        marker_idx_all = [self.get_marker_data(key) for key in marker_labels]
        marker_labels_all = [self.get_marker_labels(key) for key in marker_labels]
        
        rev_map = self.rev_id_map
         
        pids = self.proteins
        title_dict = self.get_prot_titles(pids)
        
        head = ['protein_id']
        
        for pid in pids:
            if pid.split('-')[0] in rev_map:
                second_id = True
                head.append('secondary_id')
                break
        
        else:
            second_id = False
        
        head.append('gene_name')
        head.append('description')
        head.extend(marker_heads)
        head.extend(score_heads)
        
        for    k in range(n_klasses):
            tag = klass_labels[k][:3].upper()
            head.append(f'{tag}_score_mean')

        for    k in range(n_klasses):
            tag = klass_labels[k][:3].upper()
            head.append(f'{tag}_score_std')
        
        sort_data = []
        
        for i, pid in enumerate(pids):
            
            row = [pid]
        
            if second_id:
                aids = rev_map.get(pid.split('-')[0], [])
                aids = set(x.split('.')[0] for x in aids)
                row.append(';'.join(sorted(aids)))
            
            title = title_dict.get(pid, ' : Unknown')
            gene_name, desc = title.split(' : ')
            
            row.append(gene_name)
            row.append(desc)
            
            klasses = []
            
            for m, marker_idx in enumerate(marker_idx_all):
                k = marker_idx[i]
                label = marker_labels_all[m][k]
                if m == 1 and label == 'unknown':
                    label = ''
                klasses.append(label)
            
            row += klasses
            
            p_val,novelty, completeness, p_values1, p_values2, singularity, duality = combined_scores[i]
            row += [f'{p_val:.4g}', f'{singularity:.4f}', f'{duality:.4f}']
            
            sort_key = klasses + [p_val, 1.0-singularity, 1.0-duality]
            
            
            means = [pred_data[i,:,k].mean() for k in range(n_klasses)]
            stds = [pred_data[i,:,k].std() for k in range(n_klasses)]
                
            row += [f'{x:.4f}' for x in means]
            row += [f'{x:.4f}' for x in stds]
            
            sort_data.append((sort_key, row))
        
        sort_data.sort()
        
        with open(out_file_path, 'w') as file_obj:
            write = file_obj.write
            write('\t'.join(head) + '\n')
            
            for key, row in sort_data:
                write('\t'.join(row) + '\n')
                        
        self.info(f'Wrote {n_prots:,} protein lines to {out_file_path}')
        
    
    def write_database(self, db_file_path, max_missing=0.35, min_contrib=0.05):
        
        import sqlite3
        
        aux_annos = self._aux_markers_key
        
        if 'pval' not in self.array_keys: 
            self.make_class_predictions() # prediction classes, pval, dulity etc.
            
        if self.recon_profile_key not in self.profile_keys:
            self.make_profile_predictions(max_missing) # set latent, zfill
                   
        def _chunked_execute(connection, data_rows, sql_smt, label='rows', chunk_size=1000):
 
            cursor = connection.cursor()

            n = len(data_rows)
 
            for i in range(0, n, chunk_size):
                j = min(i+chunk_size, n)
                self.info(f' .. {label} {i} - {j}', end='\r')
                cursor.executemany(sql_smt, data_rows[i:j])
 
            self.info(f' .. {label} {n}')
            cursor.close()
            connection.commit()
                
        if os.path.exists(db_file_path):
            os.unlink(db_file_path)

        connection = sqlite3.connect(db_file_path)
        connection.text_factory = str
        
        self.info('Make tables')
        cursor = connection.cursor()
        for table in DB_SCHEME:
            cursor.execute(table)
            
        cursor.close()
        connection.commit()
                
        class_labels = self.train_labels + ['UNKNOWN']
        organelles = [class_labels[x] for x in self.train_markers]

        class_labels2 = self.get_marker_labels(aux_annos) + ['UNKNOWN']
        suborganelles = [class_labels2[x] for x in self.get_marker_data(aux_annos)]

        self.info('Add Compartments')
        cursor = connection.cursor()
        codes = set([DB_ORGANELLE_CONV_DICT.get(code, code) for code in organelles + suborganelles])
        inserts = [(code, DB_ORGANELLE_INFO[code][0], DB_ORGANELLE_INFO[code][1]) for code in codes]
        cursor.executemany('INSERT INTO Compartment (code, name, color) VALUES (?,?,?)', inserts)
        cursor.close()
        connection.commit()

        self.info('Get protein alt IDs')
        rev_map = self.rev_id_map
                        
        pred_data = self.class_ensemble_preds
        n_prots, n_models, n_klasses = pred_data.shape
        
        init_profiles = self.train_profiles
        n, m = init_profiles.shape
        invalid = np.count_nonzero(np.isnan(init_profiles), axis=-1) > int(max_missing*m) # E.g. one third
     
        class_mean = pred_data.mean(axis=1)
        class_std = pred_data.std(axis=1)
        class_idx = class_mean.argsort(axis=-1)
        
        proj_srcs = ('latent','init','recon','zfill')        
        proj_names = {'latent':'DNN Latent', 'init':'Original','recon':'Reconstructed','zfill':'Zero-filled',}
        
        pids = self.proteins
        title_dict = self.get_prot_titles(pids)
        proj_2d = {}
        min_nonzero = 1.0 - max_missing        
        
        self.info('Add Data Projections')        
        projections = []
        for src in proj_srcs:
            for method in ('umap', 'pca'):
                text = f'{proj_names[src]} {method.upper()}'
                
                if method == 'umap':
                    for nn in (30,20,10):
                        key = f'{src}_{method}_{nn}'
                        self.info(f' .. {key}')
                        text = f'{proj_names[src]} {method.upper()} NN{nn}'
                        projections.append((key, text))
                        proj_2d[key] = self.get_profile_proj_2d(src, method, min_nonzero=min_nonzero, recalc=True, umap_neighbours=nn)
                
                else:
                    key = f'{src}_{method}'
                    text = f'{proj_names[src]} {method.upper()}'
                    projections.append((key, text))
                    self.info(f' .. {key}')
                    proj_2d[key] = self.get_profile_proj_2d(src, method, min_nonzero=min_nonzero, recalc=True)
                
        cursor = connection.cursor()
        cursor.executemany('INSERT INTO DataProjection (code, name) VALUES (?,?)', projections)
        cursor.close()
        connection.commit()

        class_heads = [DB_ORGANELLE_CONV_DICT.get(x, x) for x in class_labels]
        class_head_idx = list(range(len(class_heads)))
                 
        protein_inserts = []
        protein_score_inserts = []
        protein_coord_inserts = []
        done = set()
        
        pvals = self.get_array_data('pval')
        pvals_single = self.get_array_data('pval1')
        pvals_dual = self.get_array_data('pval2')
        novelty = self.get_array_data('novelty')
        completeness = self.get_array_data('completeness')
        singleness = self.get_array_data('singleness')
        predictions1 = self.get_marker_data('prediction')
        predictions2 = self.get_marker_data('prediction2')
        
        for i, pid in enumerate(pids):
            if invalid[i]:
                continue
            
            if pid.startswith('cRAP'):
                continue
            
            if pid in done:
                self.warn(f's{pid} repeats')
                continue
            else:
                done.add(pid)
            
            alt_ids = rev_map.get(pid.split('-')[0], [])
            alt_ids = set(x.split('.')[0] for x in alt_ids)
            alt_ids = ';'.join(sorted(alt_ids))
                        
            title = title_dict.get(pid, f'{pid} : Unknown')
            gene_name, description = title.split(' : ')

            train_organelle = DB_ORGANELLE_CONV_DICT.get(organelles[i], organelles[i])
            suborganelle = DB_ORGANELLE_CONV_DICT.get(suborganelles[i], suborganelles[i])
                        
            likely_single = 'YES' if float(singleness[i]) > 0.9 else 'NO'
            
            k1 = predictions1[i]
            k2 = predictions2[i]
            
            pred_class1 = DB_ORGANELLE_CONV_DICT.get(class_labels[k1], class_labels[k1])
            pred_class2 = DB_ORGANELLE_CONV_DICT.get(class_labels[k2], class_labels[k2])
            pred_class3 = ''
            
            pred_anno = []
            for k in range(n_klasses):
                mn = class_mean[i,k]
                
                if mn >= min_contrib:
                    lbl = class_labels[k]
                    if mn < 0.1:
                        lbl = f'{lbl.lower()}?'
                    elif mn < 0.2:
                        lbl = lbl.lower()
                        
                    pred_anno.append((mn, lbl))
            
            pred_text = '+'.join([x[1] for x in sorted(pred_anno, reverse=True)])
            protein_inserts.append((pid, alt_ids, description, gene_name, train_organelle,
                                    suborganelle, singleness[i], likely_single, pred_class1,
                                    pred_class2, pred_class3, pred_text, pvals[i], pvals_single[i],
                                    pvals_dual[i], completeness[i], novelty[i]))    
            
            for k, klass in enumerate(class_heads):
                protein_score_inserts.append((klass, pid, float(class_mean[i,k]), float(class_std[i,k])))
        
            for key in proj_2d:
                x, y = proj_2d[key][i]
                protein_coord_inserts.append((key, pid, float(x), float(y)))

        sql_smt = 'INSERT INTO Protein (pid, alt_ids, description, gene_name, train_organelle, suborganelle, singleness, likely_single, pred_class1, pred_class2, pred_class3, pred_text, p_val, p_val_single, p_val_dual, completeness, novelty) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)'
        _chunked_execute(connection, protein_inserts, sql_smt) 
        self.info(f'Stored {len(protein_inserts):,} proteins to {db_file_path}')

        sql_smt = 'INSERT INTO CompartmentScore (compartment, protein, score, score_std) VALUES (?,?,?,?)'
        _chunked_execute(connection, protein_score_inserts, sql_smt) 
        self.info(f'Stored {len(protein_score_inserts):,} scores to {db_file_path}')

        sql_smt = 'INSERT INTO DataCoord (projection, protein, x, y) VALUES (?,?,?,?)'
        _chunked_execute(connection, protein_coord_inserts, sql_smt) 
        self.info(f'Stored {len(protein_coord_inserts):,} coord pairs to {db_file_path}')
        
        self.info('Done')
        connection.close()            

 
    def get_train_test_data(self, profile_label=None, marker_label=None, n_chunks=5, max_nan=10): 
        
        if marker_label:
            marker_data = self.get_marker_data(marker_label)
        else:
            marker_data = self.train_markers
        
        if profile_label:
            profile_data = self.get_profile_data(profile_label)
        else:
            profile_data = self.train_profiles
        
        valid = np.count_nonzero(np.isnan(profile_data), axis=-1) <= max_nan
        
        profile_data = profile_data[valid]

        if marker_data is not None:
            marker_data = marker_data[valid]
        
        profile_data = np.nan_to_num(profile_data)
        
        idx = np.arange(len(profile_data))
        
        np.random.shuffle(idx)

        idx_chunks = np.array_split(idx, n_chunks)
        
        return valid, idx_chunks, profile_data, marker_data
        
    
    def prune_markers(self, in_marker_label, out_marker_label, fixed_classes=None, sparse_classes=None,
                      n_neighbors=11, n_rounds=1, corr_min=0.9):
        
        from scipy.spatial import distance
        from sklearn.neighbors import KNeighborsClassifier
        
        if not fixed_classes:
                fixed_classes = []
        
        if not sparse_classes:
                sparse_classes = []
         
        marker_idx = self.get_marker_data(in_marker_label)
        profile_data = np.nan_to_num(self.train_profiles)
        marker_klasses = self.get_marker_labels(in_marker_label)
        n, m = profile_data.shape
        
        in_valid = ~self.get_valid_mask()
        marker_idx[in_valid] = -1
         
        clear_train_class = np.zeros(n, dtype=bool)
        clear_train_class[in_valid] = True        
        
        cdist_thresh = 1.0 - corr_min
        
        for i in range(n_rounds):
             labelled = (marker_idx >= 0).nonzero()[0]
             
             knn_model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform', metric='correlation')
             knn_model.fit(profile_data[labelled], marker_idx[labelled])
             knn_idx = knn_model.predict(profile_data[labelled])
             inconsistent = knn_idx != marker_idx[labelled]
             
             clear_train_class[labelled] = inconsistent

             self.info(f'Round {i+1} : Removing {inconsistent.sum():,} of {len(labelled):,} labels from {n:,} profiles')
 
             for k, klass in enumerate(marker_klasses):
                   idx = (marker_idx == k).nonzero()[0]
 
                   if (klass in fixed_classes):
                       self.info(f' .. keeping all of class {klass} : {len(idx):,}')
                       clear_train_class[idx] = False
                       
                   else:
                       
                       if klass not in sparse_classes:
                           klass_profs = profile_data[idx]
                           n_peripheral = 0
 
                           for a, prof in enumerate(klass_profs):
                                corr_dists = distance.cdist(prof[None,:], klass_profs, metric='correlation')[0]
                                closest = np.argsort(corr_dists)[1]
 
                                if corr_dists[closest] > cdist_thresh:
                                    clear_train_class[idx[a]] = True
                                    n_peripheral += 1
 
                           n_remove = clear_train_class[idx].sum()
                           self.info(f' .. pruned class {klass} : removed {n_remove:,} of {len(idx):,} ; peripheral {n_peripheral:,}')
 
 
             marker_idx[clear_train_class] = -1
         
        marker_data_prune = np.full(n, -1, dtype=int)
        marker_data_prune[clear_train_class] = marker_idx[clear_train_class]
        
        self.set_marker_data(out_marker_label, marker_idx, marker_klasses)
        self.set_marker_data(f'pruned_{in_marker_label}', marker_data_prune, marker_klasses)

