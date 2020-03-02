import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import feather
import multiprocessing as mp
import scipy.cluster.hierarchy as spc
from scipy.spatial.distance import squareform
from collections import OrderedDict
from ateamopt.utils import utility
import matplotlib as mpl
import man_opt
import man_opt.utils as man_utils
from collections import defaultdict

#%% Utility functions

def sample_sim_matrix_for_cre(seed):
    cluster_list = []
    np.random.seed(seed)
    for cre_ in common_cre_lines:
        cluster_list.append(np.random.choice(cre_dict[cre_]))
    dist_matrix = cluster_sim_mat.loc[cluster_list,cluster_list]
    dist_matrix_cre = pd.DataFrame(dist_matrix.values,index=common_cre_lines,
                                   columns=common_cre_lines)
    return dist_matrix_cre

#%% Data paths
data_path = os.path.join(os.path.dirname(man_opt.__file__),os.pardir,'assets','aggregated_data')
rna_seq_data_path = '/data/rnaseqanalysis/shiny/facs_seq/Mm_VISp_14236_20180912'
cluster_sim_matrix_path = os.path.join(rna_seq_data_path,'cl.cor.csv')
mouse_data_filename = os.path.join(data_path,'Mouse_class_data.csv')
mouse_datatype_filename = os.path.join(data_path,'Mouse_class_datatype.csv')
cre_color_tasic16_filename = os.path.join(data_path,'cre_color_tasic16.pkl')
cre_ttype_filename = os.path.join(data_path,'cre_ttype_map.pkl')

cluster_sim_mat = pd.read_csv(cluster_sim_matrix_path)
cluster_sim_mat = cluster_sim_mat.rename(columns = {'Unnamed: 0':'Cluster'})
cluster_sim_mat = cluster_sim_mat.set_index('Cluster')

#%% Plot the t-type correlation matrix
sns.set(style='whitegrid')
fig,ax = plt.subplots(figsize=(12,10))
sns.heatmap(data = cluster_sim_mat,cmap='viridis',ax=ax)
ax.set_ylabel('')
utility.create_dirpath('figures')
fig.savefig('figures/rnaseq_cluster_corr.png',bbox_inches='tight',dpi=200)
plt.close(fig)


#%% Read model data    
mouse_data= man_utils.read_csv_with_dtype(mouse_data_filename,mouse_datatype_filename)
mouse_data = mouse_data.loc[mouse_data.hof_index == 0,]
cre_cluster = mouse_data.loc[mouse_data.hof_index == 0, ['Cell_id', 'Cre_line']]

annotation_datapath = os.path.join(rna_seq_data_path,'anno.feather')
annotation_df = feather.read_dataframe(annotation_datapath)
annotation_labels = list(annotation_df)
all_model_cre_lines = cre_cluster.Cre_line.unique().tolist()
all_rnaseq_cre_lines = annotation_df.cre_label.unique().tolist()
common_cre_lines = [cre_ for cre_ in all_model_cre_lines if cre_ in all_rnaseq_cre_lines]

cre_dict = {}
for cre_ in common_cre_lines:
    cre_dict[cre_] = annotation_df.loc[annotation_df.cre_label==cre_,
            'primary_cluster_label'].tolist()

cre_color_dict = utility.load_pickle(cre_color_tasic16_filename)

#%% Sample from the RNA-seq data to get Cre line similarity matrix

num_samples = int(1e4)

# Parallel sampling of distance matrix
pool = mp.Pool(4)
res = pool.map(sample_sim_matrix_for_cre, np.arange(num_samples))
res_array = [res_.values for res_ in res]
dist_cre_array = np.mean(res_array,axis=0)
dist_cre_df = pd.DataFrame(dist_cre_array,index=common_cre_lines,
                           columns=common_cre_lines)

fig,ax = plt.subplots(figsize=(12,10))
sns.heatmap(data = dist_cre_df,cmap='viridis',ax=ax)
fig.savefig('figures/cre_unsorted_rnaseq_cluster.png',bbox_inches='tight', dpi=200)
plt.close(fig)
res,res_array= None,None

#%% Hierarchical clustering based on correlation matrix
dissimilarity = 1 - dist_cre_array
max_d =  0.3*dissimilarity.max()
cre_linkage = spc.linkage(squareform(dissimilarity),method='ward')
cluster_ind = spc.fcluster(cre_linkage,max_d, 'distance')
sorted_idx = [dist_cre_df.columns.tolist()[i] for i in list((np.argsort(cluster_ind,
              kind='mergesort')))]
dist_cre_df_sorted = dist_cre_df.reindex(index=sorted_idx,columns=sorted_idx)


fig = plt.figure(figsize=(10,8))
ax1 = fig.add_axes([0.05,0.1,0.23,0.6])
Z1 = spc.dendrogram(cre_linkage, orientation='left',color_threshold=max_d,
                    above_threshold_color="lightgrey")
idx1 = Z1['leaves']
D = dist_cre_array[idx1,:]
D = D[:,idx1]
cre_line_sorted = [common_cre_lines[idx1_] for idx1_ in idx1]


# cluster cre-lines according to the cluster index
sorted_cre_lines = [list(dist_cre_df)[idx1_] for idx1_ in idx1]
sorted_cluster_ind = [cluster_ind[idx1_] for idx1_ in idx1]
u,ind = np.unique(sorted_cluster_ind,return_index=True)
sorted_cluster_ind_unique = u[np.argsort(ind)]

cre_tcluster_dict = dict(zip(sorted_cre_lines,sorted_cluster_ind))
cre_type_dict = defaultdict(list)
for cre_,cluster_type in cre_tcluster_dict.items():
    cre_type_dict[cluster_type].append(cre_)


cre_map_dict = {}
for _,cre_list in cre_type_dict.items():
    cre_models_grp = cre_cluster.loc[cre_cluster.Cre_line.isin(cre_list),].groupby('Cre_line').aggregate(np.size)
    cre_models_grp = cre_models_grp.sort_values('Cell_id').reset_index()
    abundant_cre = cre_models_grp.Cre_line.tolist()[-1]
    for cre_ in cre_list:
        cre_map_dict[cre_] = abundant_cre

cre_colormap_filename = os.path.join(data_path,'cre_ttype_map_expanded.pkl')
utility.save_pickle(cre_colormap_filename, cre_map_dict)



cre_type_dict_agg = {cre_:cre_type_dict[cluster_type] for cre_,cluster_type in cre_tcluster_dict.items()}
cre_ttype_dict_pruned = {}
for key,val in cre_type_dict_agg.items():
    if len(val) == 1:
        cre_ttype_dict_pruned[key] = val[0].split('-')[0]
    else:
        val_list = np.array([val_.split('-')[0] for val_ in val])
        _, idx = np.unique(val_list, return_index=True)
        val_list = val_list[np.sort(idx)]
        val_pruned = ''.join('/%s'%val_ for val_ in val_list)
        cre_ttype_dict_pruned[key] = val_pruned.split('/',1)[-1]
utility.save_pickle(cre_ttype_filename,cre_ttype_dict_pruned)


#%% Plot the clustermap

ax1.set_xticks([])
ax1.set_yticks([])
ax1.invert_yaxis()
ax1.axvline(x=max_d, c='k',lw=.5)
ax2 = fig.add_axes([0.47,0.1,0.48,0.6])

im = ax2.imshow(D, aspect='auto',cmap='viridis')
ax2.set_xticks([])
ax2.set_yticks(range(len(common_cre_lines)))
ax2.set_yticklabels(cre_line_sorted, minor=False)
ax2.yaxis.set_label_position('right')
ax2.grid(False)
axcolor = fig.add_axes([0.96,0.15,0.02,0.5])
cb=fig.colorbar(im,cax=axcolor)
cb.outline.set_visible(False)

ax4 = fig.add_axes([0.29,0.1,0.03,0.6])
n = len(cre_color_dict.values())
ax4.imshow(np.arange(n).reshape(n,1),
          cmap=mpl.colors.ListedColormap(list(cre_color_dict.values())),
          interpolation="nearest", aspect="auto")
ax4.set_yticks(np.arange(n))
ax4.grid(False)
ax4.set_yticklabels([])
ax4.set_xticks([])

fig.savefig('figures/cre_sorted_rnaseq_cluster.pdf',bbox_inches='tight')
plt.close(fig)