# A script to gather training input and labels for DNN
#   Gather proteomic profiles for each database
#   Normalise proteomic profiles
#   Add marker lists for each dataset
#   Prune markler list to make training sets

# NOTE Human U2OS data is already normalised in the TSV file

import os
from schisome import SchisomeDataSet

if not os.path.exists('datasets'):
    os.mkdir('datasets')

run_tag = 'Aug25v1'

plot_args = dict(min_nonzero=0.9, spot_size=16)


# # # # # # # # # # # # # # # # # # # # 

## ## Add Miguel's experimental lists

source_tag = 'Arabidopsis_PCSC'

profile_paths = ['profiles/Arabidposis_PCSC_R0.tsv',
                 'profiles/Arabidposis_PCSC_R1.tsv',
                 'profiles/Arabidposis_PCSC_R2.tsv',
                 'profiles/Arabidposis_PCSC_R3.tsv',
                 'profiles/Arabidposis_PCSC_R4.tsv',
                 'profiles/Arabidposis_PCSC_R5.tsv',
                 'profiles/Arabidposis_PCSC_R6.tsv']

data_path = f'datasets/{source_tag}_{run_tag}.npz'

at_dataset = SchisomeDataSet(data_path, source_tag, aux_marker_key='suborganelle')
at_dataset.add_raw_profiles(profile_paths)
at_dataset.add_markers('organelle', 'markers/Arabidopsis_organelle_Feb23.csv')
at_dataset.add_markers('suborganelle', 'markers/Arabidopsis_suborganelle_Feb23.csv')

at_dataset.normalize_profiles('init', col_norm=False)
at_dataset.prune_markers('organelle', 'training')

at_dataset.plot_umap_2d('init', ['organelle', 'training'], ['Organelle markers', 'Training'], **plot_args)


# # # # # # # # # # # # # # # # # # # # 


source_tag = 'Human_U2OS'

profile_paths = ['profiles/Human_U2OS_all.tsv']

# Curated markers from data orig publication

data_path = f'datasets/{source_tag}_{run_tag}.npz'

hs_dataset = SchisomeDataSet(data_path, source_tag, aux_marker_key='uniprot')
hs_dataset.add_raw_profiles(profile_paths)

hs_dataset.add_markers('organelle', 'markers/Human_U2OS_markers_nocplx.csv')

if not os.path.exists('markers/Human_UniProt_all.csv'):
    hs_dataset.add_uniprot_markers('uniprot')
    hs_dataset.write_markers('uniprot', 'markers/Human_UniProt_all.csv')
else:
    hs_dataset.add_markers('uniprot',   'markers/Human_UniProt_all.csv')

hs_dataset.prune_markers('organelle', 'training', save_file=None)  


hs_dataset.plot_umap_2d('init', ['organelle', 'uniprot', 'training'], ['SVM markers', 'UniProt', 'Training'], **plot_args)

# UniProt derived markers

data_path2 = f'datasets/{source_tag}_UniProt_{run_tag}.npz'

hs_dataset2 = SchisomeDataSet(data_path2, source_tag, aux_marker_key='svn_in')
hs_dataset2.add_raw_profiles([profile_paths])

hs_dataset2.add_markers('organelle', 'markers/Human_UniProt_all.csv')
hs_dataset2.add_markers('svn_in',    'markers/Human_U2OS_markers_nocplx.csv')
hs_dataset2.prune_markers('organelle', 'training', save_file=None, sparse_classes=['ENDOSOME'])  

hs_dataset2.plot_umap_2d('init', ['organelle', 'svn_in', 'training'], ['UniProt', 'SVM markers', 'Training'], **plot_args)


# # # # # # # # # # # # # # # # # # # # 


source_tag = 'Mouse_E14'

profile_paths = ['profiles/Mouse_E14_R1.csv',
                 'profiles/Mouse_E14_R2.csv']

# Curated markers from data orig publication

data_path = f'datasets/{source_tag}_{run_tag}.npz'

mm_dataset = SchisomeDataSet(data_path, source_tag, aux_marker_key='TAGM_good')
mm_dataset.add_raw_profiles(profile_paths)
mm_dataset.normalize_profiles('init', col_norm=False)
 
mm_dataset.add_markers('organelle',  'markers/Mouse_E14_markers.csv')
mm_dataset.add_markers('TAGM_good',  'markers/Mouse_E14_TAGMgood.csv')
mm_dataset.prune_markers('organelle', 'training')  

# Get Uniprot markers; save these for next time
if not os.path.exists('markers/Mouse_UniProt_all.csv'):
    mm_dataset.add_uniprot_markers('uniprot')
    mm_dataset.write_markers('uniprot', 'markers/Mouse_UniProt_all.csv')
else:
    mm_dataset.add_markers('uniprot',   'markers/Mouse_UniProt_all.csv')

mm_dataset.plot_umap_2d('init', ['organelle', 'TAGM_good', 'training'], ['TAGM markers', 'TAGM good', 'Training'], **plot_args)

# UniProt derived markers

data_path2 = f'datasets/{source_tag}_UniProt_{run_tag}.npz'

mm_dataset2 = SchisomeDataSet(data_path2, source_tag, aux_marker_key='TAGM_good')
mm_dataset2.add_raw_profiles([profile_paths])
mm_dataset2.normalize_profiles('init', col_norm=False)

mm_dataset2.add_markers('organelle',  'markers/Mouse_UniProt_all.csv')
mm_dataset2.add_markers('TAGM_good',  'markers/Mouse_E14_TAGMgood.csv')
mm_dataset2.prune_markers('organelle', 'training')  

mm_dataset2.plot_umap_2d('init', ['organelle', 'svn_in', 'training'], ['UniProt', 'TAGM good', 'Training'], **plot_args)
