from ateamopt.utils import utility
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
import numpy as np
import glob
import man_opt

#%% Get the data
data_path = os.path.join(os.path.dirname(man_opt.__file__),os.pardir,'assets','aggregated_data')
HoF_obj_train_path = '/allen/aibs/mat/ateam_shared/Mouse_Model_Fit_Metrics/*/hof_obj_train.pkl'
HoF_obj_train_filelist = glob.glob(HoF_obj_train_path)


#%%  Create heatmap of the feature errors

obj_dict_train_list = [] 
feat_reject_list = ['Spikecount','depol_block','check_AISInitiation']
max_feat_count = {}
feature_mat_list = []
feature_mat_dict_all = defaultdict(list)

feature_heatmap_filename = os.path.join('figures','feature_score_comparison.pdf')
utility.create_filepath(feature_heatmap_filename)

for obj_file in HoF_obj_train_filelist:
    obj_dict_train = utility.load_pickle(obj_file)[0]
    cell_id = obj_file.split('/')[-2]
    temp_dict = defaultdict(list)
    for obj_key,obj_val in obj_dict_train.items():
        feat = obj_key.split('.')[-1] 
        feat = feat if feat != 'AP_amplitude_from_voltagebase' else 'AP_amplitude'
        if feat not in feat_reject_list:
            temp_dict[feat].append(obj_val)
            feature_mat_dict_all[feat].append(obj_val)
            feature_mat_dict_all['Cell_id'].append(cell_id)
    temp_dict = {key:np.mean(val) for key,val in temp_dict.items()}
    
    max_feat = max(temp_dict, key=temp_dict.get)

    temp_dict['max_feature'] = max_feat
    temp_dict['max_feature_val'] = temp_dict[max_feat]
    
    if max_feat in max_feat_count.keys():
        max_feat_count[max_feat] += 1
    else:
        max_feat_count[max_feat] = 1
        
    temp_dict['Cell_id'] = cell_id
    feature_mat_list.append(temp_dict)
    
feature_mat_df = pd.DataFrame(feature_mat_list,index=None)   
feature_mat_df_lite = feature_mat_df.loc[:,feature_mat_df.columns.difference(['max_feature','max_feature_val'])]

feature_mat_df_lite.to_csv(os.path.join(data_path,'Feature_performance.csv'),index=False)
feature_mat_df = feature_mat_df.rename(columns={'AP_amplitude_from_voltagebase':'AP_amplitude'})
max_feat_count_sorted = sorted(max_feat_count,
                key = lambda x:max_feat_count[x],reverse=True) 

feature_mat_all_list = []
for feat_ in max_feat_count_sorted:
    feature_mat_all_list.append(pd.DataFrame({'feature': [feat_]*len(feature_mat_dict_all[feat_]),
                    'z_score' : feature_mat_dict_all[feat_]}))
feature_all_df = pd.concat(feature_mat_all_list)
feature_all_df['Cell_id'] =  feature_mat_dict_all['Cell_id']       
max_feat_count_sorted_dict = {feat:ii for ii,feat in enumerate(max_feat_count_sorted)}
feature_mat_df['max_feature_index'] = feature_mat_df['max_feature'].apply(lambda x:max_feat_count_sorted_dict[x])

feature_mat_df = feature_mat_df.sort_values(by=['max_feature_index','max_feature_val'])
        
feature_mat_im = feature_mat_df[max_feat_count_sorted].reset_index(drop=True)
feature_mat_im = feature_mat_im.T
feature_mat_im = feature_mat_im.iloc[::-1] # Reverse to have best performing features on top
#%% Plot the figure

sns.set(style='whitegrid')
fig,ax = plt.subplots(1,2,figsize=(10,5),
            gridspec_kw={'width_ratios': [10, .5],
                        'wspace':.01})
axis_fontsize = 12
tick_fontsize = 12
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
#    cmap = sns.cubehelix_palette(as_cmap=True)
ax[0] = sns.heatmap(feature_mat_im,cmap=cmap,alpha=0.8,
                    ax = ax[0],cbar_kws={'label':'z-score'},
                    vmin=0,vmax=10)
xticklabels = ax[0].get_xticklabels()
ax[0].set_xticklabels(labels=
                    ['']*len(xticklabels))
plt.setp(ax[0].get_yticklabels(),fontsize=tick_fontsize)
ax[0].set_xlabel('Models',fontsize=axis_fontsize)

feature_median = np.nanmedian(feature_mat_im.values,axis=1)
feature_median = feature_median.reshape((len(feature_median),1))
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
#    cmap = sns.cubehelix_palette(as_cmap=True)
ax[1] = sns.heatmap(feature_median,
                    cmap=cmap,alpha=0.8,
                    ax = ax[1],cbar=False,
                    vmin=0,vmax=10)
xticklabels = ax[1].get_xticklabels()
yticklabels = ax[1].get_yticklabels()
ax[1].set_xticklabels(labels=
                    ['']*len(yticklabels))
ax[1].set_yticklabels(labels=
                    ['']*len(xticklabels))
ax[1].set_xlabel('Median',fontsize=axis_fontsize)
fig.savefig(feature_heatmap_filename, bbox_inches='tight')
plt.close(fig)

