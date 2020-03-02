import pandas as pd
import seaborn as sns
from ateamopt.utils import utility
import matplotlib.pyplot as plt
import os
import man_opt.utils as man_utils
import numpy as np
from multiprocessing import Pool
import multiprocessing



def calc_param_dist(param_df):
    '''
    Calculates the dispersion among the hall of fame indices for a single cell
    and the total dispersion
    '''
    param_df = param_df.sort_values('hof_index')
    param_df = param_df.dropna(axis=1,how='all')
    nunique = param_df.apply(pd.Series.nunique)
    # For inh cells in a couple cases the apical dendrite parameters were not removed
    # which generated identical values across the hall of fame indices for these apical
    # parameters
    cols_to_drop = nunique[nunique == 1].index.tolist() + ['hof_index']
    
    param_values = param_df.drop(labels=cols_to_drop,axis=1).values
   
    param_values = np.transpose(param_values)
    param_mean = np.mean(param_values,axis=1)
    vec_len = param_mean.shape[0]
    
    # Normalized with means since we don't want the following behaviors
    # a MinMaxScaler() will restrict the min,max values to [0,1] across all dimensions
    # a StandardScaler() will restrict the standard deviation in each dimension to 1 
    param_values_sub = (param_values - param_mean[:,None])/param_mean[:,None]
    sub_norm_vec = np.linalg.norm(param_values_sub,axis=0)
    sub_norm_total = np.sum(sub_norm_vec)/vec_len
    sub_norm_individual = sub_norm_vec/vec_len
    dist_array = np.append(sub_norm_individual,sub_norm_total)
    return dist_array

def calc_param_separation(hof_param_df):
    '''
    Calculates the dispersion for all cells by parallelizing the dispersion calculation
    '''
    hof_param_df = hof_param_df.dropna(how='any',axis=1)
    cell_ids = hof_param_df.Cell_id.unique()
    param_df_list = [hof_param_df.loc[hof_param_df.Cell_id == cell_id,]\
                     for cell_id in cell_ids]
    p = Pool(multiprocessing.cpu_count())
    param_dist_list = p.map(calc_param_dist,param_df_list)
    p.close()
    p.join()
    return param_dist_list

# Data path
data_path = os.path.join(os.getcwd(),os.pardir,os.pardir,'assets','aggregated_data')
mouse_data_filename = os.path.join(data_path,'Mouse_class_data.csv')
mouse_datatype_filename = os.path.join(data_path,'Mouse_class_datatype.csv')
param_data_filename = os.path.join(data_path,'allactive_params.csv')
param_datatype_filename = os.path.join(data_path,'allactive_params_datatype.csv')


mouse_data_df = man_utils.read_csv_with_dtype(mouse_data_filename,mouse_datatype_filename)
hof_param_data = man_utils.read_csv_with_dtype(param_data_filename,param_datatype_filename)
hof_param_fields = list(hof_param_data)
param_fields = hof_param_fields + ['Dendrite_type']
hof_param_type_df = mouse_data_df.loc[:,param_fields] 
hof_param_type_df = hof_param_type_df.sort_values(['Cell_id','hof_index'])
hof_param_spiny = hof_param_type_df.loc[hof_param_type_df.Dendrite_type=='spiny',hof_param_fields]
num_spiny_cells = int(hof_param_spiny.shape[0]/40)
hof_param_aspiny = hof_param_type_df.loc[hof_param_type_df.Dendrite_type=='aspiny',hof_param_fields]
hof_param_df = pd.concat([hof_param_spiny,hof_param_aspiny])
hof_cell_ids_exc = hof_param_spiny['Cell_id'].unique()
hof_cell_ids_inh = hof_param_aspiny['Cell_id'].unique()
hof_cell_ids_dist = np.concatenate((hof_cell_ids_exc,hof_cell_ids_inh))
num_cells = int(hof_param_df.shape[0]/40)

param_dist_list = calc_param_separation(hof_param_df)
param_dist_arr = np.array(param_dist_list).T

param_dist_arr_hof = param_dist_arr[:-1,:]
param_dist_sum = param_dist_arr[-1,:]

dist_max_exc_idx = np.argmax(param_dist_sum[:num_spiny_cells])
dist_min_exc_idx = np.argmin(param_dist_sum[:num_spiny_cells])
dist_max_inh_idx = num_spiny_cells+ np.argmax(param_dist_sum[num_spiny_cells:])
dist_min_inh_idx = num_spiny_cells +np.argmin(param_dist_sum[num_spiny_cells:])

dist_indices = [dist_max_exc_idx,dist_min_exc_idx,dist_max_inh_idx,
                dist_min_inh_idx]
color_dist_list =  sns.color_palette('Set1',len(dist_indices))
y_dist_indices = [-1.75]*len(dist_indices)

sns.set(style='whitegrid')
fig,(ax1,ax2) = plt.subplots(nrows=2, 
                 sharex=True,gridspec_kw={'height_ratios':[1,5]})
ax1.bar(np.arange(0,len(param_dist_sum)),param_dist_sum,linewidth=0,
        color= 'grey')
ax1.set_ylabel('Cluster\n dispersion',fontsize=10)
plt.setp(ax1.get_yticklabels(),fontsize=10)
ax1.grid(False)
ax1.set_frame_on(False)

im = ax2.imshow(param_dist_arr_hof,cmap='viridis',aspect="auto")
ax2.scatter(dist_indices,y_dist_indices,color = color_dist_list,s=50)
y_lims = ax2.get_ylim()

h =1
ax2.plot([num_spiny_cells-h,num_spiny_cells-h],[y_lims[0],-.5] ,
          lw=.5,color='k',alpha=1)
for dist_ind in dist_indices:
    ax2.plot([dist_ind-h,dist_ind-h],[y_lims[0],-.5] ,
          lw=.5, linestyle= '--',color='w',alpha=1)
    ax2.plot([dist_ind+h,dist_ind+h],[y_lims[0],-.5] ,
          lw=.5,linestyle= '--',color='w',alpha=1)
    
ax2.set_ylim([-3.5,39.5])
ax2.grid(False)
ax2.invert_yaxis()
ax2.set_frame_on(False)
ax2.set_xlabel('Models',fontsize=10)
ax2.set_ylabel('Hall of Fame index',fontsize=10)
plt.setp(ax2.get_xticklabels(),fontsize=10)
plt.setp(ax2.get_yticklabels(),fontsize=10)
cbar=fig.colorbar(im,ax=ax2,orientation="horizontal",fraction=0.05, 
                  pad=0.25,ticks=[0.25,.5,.75])
cbar.ax.set_xlabel('Normalized distance',fontsize=10)
cbar.ax.tick_params(axis=u'both', which=u'both',length=0,labelsize=10)
cbar.outline.set_visible(False)
fig.subplots_adjust(hspace=.05)
fig.savefig('figures/Dist_param_space.svg',bbox_inches='tight')
plt.close(fig)