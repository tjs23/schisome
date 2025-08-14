import sys, os, glob
from schisome import SchisomeDataSet

run_tag = 'Aug25v1'
data_paths = glob.glob(f'datasets/*_{run_tag}.npz')
umap_titles = ['Input', 'Zero-filled', 'Latent']

for data_path in data_paths:
  
  # # # # # # # # 
  
  data_set = SchisomeDataSet(data_path)  
 
  tag = data_set.source_tag
  
  latent_key = data_set.latent_profile_key
  zfill_key = data_set.zfill_profile_key
  train_key = data_set.train_profile_key
  
  data_set.save_pruned_table(tsv_path=f'{tag}_pruned_list.tsv')
 
  data_set.plot_overview()
  
  data_set.plot_dual_localisation_overview(save_path=f'plots/{tag}_dual_matrix.png', tsv_path=f'{tag}_duals_list.tsv')
  
  data_set.plot_pr_curve()  

  data_set.plot_umap_2d([zfill_key, latent_key], 'pval', ['Zero-reconstructed','Latent'],
                        title=f'UMAP Zero-reconstructed p-value',
                        save_path=f'plots/{tag}_pvalue.png')

  data_set.plot_umap_2d([zfill_key], 'dual', ['Zero-reconstructed'],
                        title=f'UMAP Zero-reconstructed duality',
                        save_path=f'plots/{tag}_duality.png')

  data_set.plot_umap_2d([zfill_key], 'nzeros', ['Zero-reconstructed'],
                        title=f'UMAP Original zero-content',
                        save_path=f'plots/{tag}_zerocount.png')

  data_set.plot_umap_2d(train_key, ['pruned_organelle','training'], ['Removed','Retained'],
                        title=f'UMAP Training Marker Cull',
                        save_path=f'plots/{tag}_marker_prune.png')
                        
  umap_prof_keys = [train_key, zfill_key, latent_key]
  
  for marker_key in (data_set.train_markers_key, 'predictions', data_set.aux_markers_key):
      data_set.plot_umap_2d(umap_prof_keys, marker_key, umap_titles,
                            title=f'UMAP {tag} {marker_key} classes',
                            save_path=f'plots/{tag}_{marker_key}.png')

  for prof_key in umap_prof_keys:
      data_set.plot_dual_proj_2d(prof_key, title=f'UMAP {prof_key} {tag} : Dual localisation',
                                 save_path=f'plots/{tag}_dual_loc_{prof_key}_{{}}.png')
 
  data_set.plot_contingency_table(data_set.train_markers_key,
                                  save_path=f'plots/{tag}_confusion_matrix_train.pdf',
                                  marker_title=f'{tag} : Train Class')
                                  
  data_set.plot_contingency_table(data_set.raw_markers_key,
                                  save_path=f'plots/{tag}_confusion_matrix_unfiltered.pdf',
                                  marker_title=f'{tag} : Unfiltered Marker Class')
  
  if tag.startswith('Arabidposis'):
      data_set.plot_reconstruction()  
      data_set.plot_l2_loss_distrib()
  
      # Plots illustrating mixing of organelle pairs
 
      data_set.plot_ave_profiles('prediction',  save_paths=f'plots/{tag}_ave_pred_profile_{{}}.pdf')
      data_set.plot_ave_profiles(data_set.train_markers_key, save_paths=f'plots/{tag}_ave_train_profile_{{}}.pdf')
 
      save_paths = f'plots/{tag}_mix_ave_prof_{{}}_{{}}.pdf'
      data_set.plot_mixed_ave_profiles('CYTOSOL', 'MITOCHONDRION', save_paths=save_paths)
      data_set.plot_mixed_ave_profiles('CHLOROPLAST', 'MITOCHONDRION', save_paths=save_paths)
      data_set.plot_mixed_ave_profiles('GOLGI', 'ER', save_paths=save_paths)
      data_set.plot_mixed_ave_profiles('GOLGI', 'TGN', save_paths=save_paths)
      data_set.plot_mixed_ave_profiles('PM', 'TGN', save_paths=save_paths)
      data_set.plot_mixed_ave_profiles('PM', 'ER', save_paths=save_paths)
      data_set.plot_mixed_ave_profiles('CYTOSOL', 'CHLOROPLAST', save_paths=save_paths)
      data_set.plot_mixed_ave_profiles('CYTOSOL', 'NUCLEUS', save_paths=save_paths)

  if tag.startswith('Mouse'):
      data_set.plot_prediction_scatter(save_paths='plots/{tag}_DNNscorescatter_TAGEMexemplar.pdf',
                                       protein_ids=['G5E870','Q924C1','Q9WUA2','Q8VDR9'])
     


  
  
