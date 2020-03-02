import pandas as pd
import os
from ateamopt.utils import utility
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection
import numpy as np
import man_opt
import man_opt.utils as man_utils


def get_feature_vec(feature_dict,select_features,select_stims,cell_id,hof_index,cre):
    feature_vec_dict = {}
    spikecount_arr = []
    for ii,stim in enumerate(select_stims):
        spikecount_arr.append(feature_dict['%s.soma.Spikecount'%stim])
        for feature in select_features:
            try:
                feature_vec_dict['%s_%s'%(feature,ii)] = feature_dict['%s.soma.%s'%(stim,feature)]
            except:
                feature_vec_dict['%s_%s'%(feature,ii)] = None
    feature_vec_dict['Cell_id'],feature_vec_dict['hof_index'] = cell_id,hof_index
    feature_vec_dict['Cre_line'] = cre
    return feature_vec_dict,spikecount_arr

def get_fi_slope(stim_fi,spike_fi):
    x = np.array(stim_fi)
    y = np.array(spike_fi)
    A = np.vstack([x, np.ones_like(x)]).T
    m, _ = np.linalg.lstsq(A, y,rcond=None)[0]
    return m



def annotate_sig_level(var_levels,hue_levels,hue_var,sig_group,sig_group_var,
                       plot_data,plot_data_x,plot_data_y,ax):
    y_sig_dict = {}
    for ii,var_level in enumerate(var_levels):
        whisk_max = -np.inf
        for hue_ in hue_levels:
            data = plot_data.loc[(plot_data[plot_data_x] == var_level) &\
                                 (plot_data[hue_var] == hue_),plot_data_y]
            data = data.dropna(how='any',axis=0)
            iqr = np.percentile(data,75) -np.percentile(data,25)
            whisk_ = np.percentile(data,75)+ 1.5 *iqr
            whisk_max = whisk_ if whisk_ > whisk_max else whisk_max
       
        y_sig_dict[var_level] = whisk_max
    y_sig_max = np.max(list(y_sig_dict.values()))
    
    for y_sig_key,y_sig_val in y_sig_dict.items():
        if y_sig_val != y_sig_max:
            y_sig_dict[y_sig_key] = (y_sig_val/y_sig_max + .4)*y_sig_max 
    
    mid_hue_index = len(hue_levels)/2.0

    bar_gap = 0.1
    bar_width = (1-2*bar_gap)/len(hue_levels)
    for param_,group in sig_group:
        y_sig = y_sig_dict[param_]
        h= .04*abs(y_sig)
        for _,row_group in group.iterrows():
            cre_x,cre_y =row_group[sig_group_var].split('~')
            cre_x_idx = hue_levels.index(cre_x)
            cre_y_idx = hue_levels.index(cre_y)
            
            x_center = var_levels.index(param_)
            x_drift_x = abs(cre_x_idx-mid_hue_index+0.5)*bar_width
            x_drift_y = abs(cre_y_idx-mid_hue_index+0.5)*bar_width
            if cre_x_idx < cre_y_idx:
                x1,x2 = x_center - x_drift_x,x_center + x_drift_y
            else:
                x1,x2 = x_center +x_drift_x,x_center - x_drift_y
            ax.plot([x1, x1, x2, x2], [y_sig, y_sig+h, y_sig+h, y_sig], lw=1, 
                     c='k') 

            ax.text((x1+x2)*.5, y_sig+h, row_group['sig_level'], ha='center', 
                    va='bottom', color='k')
            y_sig += 5*h
            
    return ax


    
hof_num = 1

inh_lines = ["Htr3a-Cre_NO152","Sst-IRES-Cre","Pvalb-IRES-Cre"]
data_path = os.path.join(os.path.dirname(man_opt.__file__),os.pardir,'assets','aggregated_data')
mouse_data_filename = os.path.join(data_path,'Mouse_class_data.csv')
mouse_datatype_filename = os.path.join(data_path,'Mouse_class_datatype.csv')
sdk_data_filename = os.path.join(data_path,'sdk.csv')
sdk_datatype_filename = os.path.join(data_path,'sdk_datatype.csv')
cre_coloring_filename = os.path.join(data_path,'cre_color_tasic16.pkl')
filtered_me_inh_cells_filename = os.path.join(data_path,'filtered_me_inh_cells.pkl')


mouse_data = man_utils.read_csv_with_dtype(mouse_data_filename,mouse_datatype_filename)
mouse_data = mouse_data.loc[mouse_data.hof_index<hof_num,]
mouse_data = mouse_data.loc[mouse_data.Cre_line.isin(inh_lines),['Cell_id','Cre_line','hof_index']]
cre_color_dict = utility.load_pickle(cre_coloring_filename)
filtered_me_inh_cells =  utility.load_pickle(filtered_me_inh_cells_filename)

sdk_data= man_utils.read_csv_with_dtype(sdk_data_filename,sdk_datatype_filename)
sdk_data= sdk_data.rename(columns={'line_name':'Cre_line','ef__threshold_i_long_square':'rheobase'})
sdk_data = sdk_data.loc[sdk_data.Cre_line.isin(inh_lines),['Cell_id','rheobase']]
mouse_data = pd.merge(mouse_data,sdk_data,on='Cell_id')

cell_ids = list(set(mouse_data.Cell_id.tolist()))
efel_feature_path = 'eFEL_features'

spiking_features_df_list = []

spiking_features = ['AP_amplitude_from_voltagebase','AP1_amp','APlast_amp',
                    "AP_width",'AHP_depth','mean_frequency',
                    'Spikecount','inv_first_ISI','voltage_base']
num_supthresh_stims = 4
metadata_list = ['Cell_id','hof_index','Cre_line']


for idx,cell_id in enumerate(cell_ids):
    cell_efeatures_dir = os.path.join(efel_feature_path,cell_id)
    cell_protocols_filename = os.path.join(cell_efeatures_dir,'protocols.json')
    stimmap_data_path = os.path.join(efel_feature_path,cell_id,'StimMapReps.csv')    
    stimmap_data = pd.read_csv(stimmap_data_path, sep='\s*,\s*',
                                   header=0, encoding='ascii', engine='python')
    stimmap_data['Amplitude_Start'] *= 1e12
    stimmap_data['Amplitude_Start'] = round(stimmap_data['Amplitude_Start'],2)
    stimmap_data['Amplitude_End'] *= 1e12
    cell_protocols= utility.load_json(cell_protocols_filename)
    
    stimmap_data = stimmap_data.sort_values(by='Amplitude_Start')
    
    cre = mouse_data.loc[mouse_data.Cell_id == cell_id,'Cre_line'].values[0]
    rheobase= sdk_data.loc[sdk_data.Cell_id == cell_id,'rheobase'].values[0]
    
    spiking_stims = stimmap_data.loc[stimmap_data['Amplitude_Start']>=rheobase,
                                             'DistinctID'].tolist()
    spiking_stim_amp = stimmap_data.loc[stimmap_data['Amplitude_Start']>=rheobase,
                                             'Amplitude_Start'].tolist()
    for hof_index in range(hof_num):
        model_features_fileaname = os.path.join('model_features','%s_features.json'%(cell_id))
        model_features = utility.load_json(model_features_fileaname)
        spiking_feature_dict = {stim_feat_name:feat_val for stim_feat_name,feat_val
                   in model_features.items() if stim_feat_name.split('.')[0] in spiking_stims and 
                    stim_feat_name.split('.')[-1] in spiking_features}
        
        spiking_feature_dict_keys = list(spiking_feature_dict.keys())
        feature_vec_dict,spikecount_arr=get_feature_vec(spiking_feature_dict,spiking_features,
                                                   spiking_stims,cell_id,hof_index,cre)
        fi_slope = get_fi_slope(spiking_stim_amp,spikecount_arr) 
        feature_vec_dict['fi_slope'] = fi_slope
        spiking_features_df_list.append(feature_vec_dict)
        

spiking_features_df = pd.DataFrame(spiking_features_df_list)

indexed_spiking_features = [feat+'_'+str(ii) for ii in range(num_supthresh_stims) for feat in spiking_features]
indexed_spiking_features.append('fi_slope')

spiking_features_df = spiking_features_df.iloc[:,spiking_features_df.columns.get_indexer(
        indexed_spiking_features+metadata_list).tolist()]

spiking_features_df.to_csv('spiking_data_model.csv',index=True)
spiking_data = spiking_features_df.dropna(how='any',axis=0)

palette = {cre:cre_color_dict[cre] for cre in inh_lines}


#%% Compare features between Cre-lines

stim_index = 3 # 60pA above rheobase
select_spiking_features = ['AP_amplitude_from_voltagebase','AP_width',
                           'AHP_depth','mean_frequency','fi_slope']

select_spiking_features = [feat_+'_%s'%stim_index if feat_ != 'fi_slope'
                           else feat_ for feat_ in select_spiking_features] 

rename_dict = {'AP_width_3':'AP_width','AHP_depth_3':'AHP_depth',
                                  'mean_frequency_3':'spike frequency',
                                  'fi_slope':'fi_slope'}
all_renamed_feats = list(rename_dict.values())
select_spiking_df = spiking_features_df.loc[:,select_spiking_features+['Cell_id','Cre_line']]
select_spiking_df = select_spiking_df.rename(columns=rename_dict)
unit_list = ['$mV$','$ms$','$mV$','$Hz$','$Hz/pA$']

# Significance testing
diff_ephys_df = []
p_val_list = []
for efeat in all_renamed_feats:
    for comb in combinations(inh_lines, 2):  # 2 for pairs, 3 for triplets, etc
        cre_x,cre_y = comb
        cre_x_efeat = select_spiking_df.loc[select_spiking_df.Cre_line == cre_x,efeat].values
        cre_y_efeat = select_spiking_df.loc[select_spiking_df.Cre_line == cre_y,efeat].values
        _,p_val = mannwhitneyu(cre_x_efeat,cre_y_efeat,alternative='two-sided')
        comp_type = '%s~%s'%(cre_x,cre_y)
        sig_dict = {'comp_type' : comp_type,
            'feature': efeat}
        diff_ephys_df.append(sig_dict)
        p_val_list.append(p_val)
        
_,p_val_corrected = fdrcorrection(p_val_list)

diff_ephys_df = pd.DataFrame(diff_ephys_df)
diff_ephys_df['p_val'] = p_val_corrected
diff_ephys_df['sig_level'] = diff_ephys_df['p_val'].apply(lambda x: man_utils.pval_to_sig(x))

spiking_melt_df = pd.melt(select_spiking_df,id_vars=['Cell_id','Cre_line'],
                          value_vars=all_renamed_feats,var_name='features',value_name='value')

# filtered ME cells
filtered_me_inh_cells =  utility.load_pickle(filtered_me_inh_cells_filename)
spiking_melt_df = spiking_melt_df.loc[spiking_melt_df.Cell_id.isin(filtered_me_inh_cells),]

ylim_list = [1.6, 60,220, 2.0]

sns.set(style='whitegrid')
fig,ax = plt.subplots(1,len(all_renamed_feats),sharey=False,figsize=(12,3))
for ii,feat_ in enumerate(all_renamed_feats):
    data = spiking_melt_df.loc[spiking_melt_df.features == feat_,]
    
    median_data = data.groupby(["Cre_line",'features']).value.median()
    quantile_data = data.groupby(["Cre_line",'features']).value.quantile([.25,.75]).unstack()
    quantile_data['iqr'] = quantile_data[0.75]-quantile_data[0.25]
    print('%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Model Median data')
    print(median_data)
    print('Model Quantile data')
    print(quantile_data.loc[:,'iqr'])
    
    sig_df = diff_ephys_df.loc[diff_ephys_df.feature == feat_,:]
    sig_vars = sig_df.feature.tolist()
    ephys_sig_group = sig_df.groupby('feature')
    
    ax[ii] = sns.boxplot(x='features',y='value',data=data,hue='Cre_line',
          hue_order=inh_lines,ax=ax[ii],palette=palette,linewidth=1,showfliers=False)
    
    for patch in ax[ii].artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .7))
    
    ax[ii] = annotate_sig_level(sig_vars,inh_lines,'Cre_line',
                         ephys_sig_group,'comp_type',
                         data,'features','value',ax[ii])
    ax[ii] = sns.stripplot(x='features',y='value',data=data,hue='Cre_line',
          hue_order=inh_lines,ax=ax[ii],palette=palette,alpha=0.5,dodge=True)
    ax[ii].grid(False)
    sns.despine(ax=ax[ii])
    ax[ii].get_legend().remove()
    ax[ii].set_xlabel('')
    ax[ii].set_ylabel('')
    ax[ii].set_ylim([0,ylim_list[ii]])
fig.subplots_adjust(wspace=.45)
fig.savefig('figures/spike_features_model_comparison_inh.svg',bbox_inches='tight')
plt.close(fig)
