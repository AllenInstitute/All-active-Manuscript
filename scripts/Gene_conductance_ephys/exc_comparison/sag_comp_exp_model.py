import pandas as pd
from ateamopt.utils import utility
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection
import numpy as np
import dabest
import man_opt
import os
import man_opt.utils as man_utils
import matplotlib as mpl


# Data paths
data_path = os.path.join(os.path.dirname(man_opt.__file__),os.pardir,'assets','aggregated_data')
cre_coloring_filename = os.path.join(data_path,'cre_color_tasic16.pkl')
sag_features_exp_filename = os.path.join(data_path,'sag_data_exp.csv')
sag_features_model_filename = os.path.join(data_path,'sag_data_model.csv')
filtered_me_exc_cell_list_path = os.path.join(data_path,'filtered_me_exc_cells.pkl')
perturbed_Nr5like_feature_file = os.path.join(data_path,'sag_data_perturbed_Nr5like.csv')
perturbed_Rbp4like_feature_file = os.path.join(data_path,'sag_data_perturbed_Rbp4like.csv')


# Read the data
exc_lines = ["Nr5a1-Cre","Rbp4-Cre_KL100"]
cre_color_dict = utility.load_pickle(cre_coloring_filename)
cre_color_dict["Rbp4-Cre_KL100"] = mpl.colors.to_rgb('#008033') # Only for better contrast
filtered_me_exc_cells = utility.load_pickle(filtered_me_exc_cell_list_path)

# All experiment
sag_features_exp = pd.read_csv(sag_features_exp_filename,index_col=0,dtype={'Cell_id':str})
sag_features_exp = sag_features_exp.loc[sag_features_exp.Cell_id.isin(filtered_me_exc_cells),]
sag_features_exp['type'] = 'Exp'

# Models
sag_features_model = pd.read_csv(sag_features_model_filename,index_col=0,dtype={'Cell_id':str})
sag_features_model = sag_features_model.loc[sag_features_model.Cell_id.isin(filtered_me_exc_cells)]
sag_features_model['type'] = 'Model'
model_cell_ids = sag_features_model.Cell_id.tolist()

feat_list = [feat_ for feat_ in sag_features_exp.columns if feat_ not in ['Cell_id',
                                      'hof_index','Cre_line','type','sample_index']]

sag_perturbed_Rbp4like = pd.read_csv(perturbed_Rbp4like_feature_file,index_col=0)
# Calculate the mean over repetitions of the perturbed models
sag_perturbed_Rbp4like = sag_perturbed_Rbp4like.groupby('Cell_id')[feat_list].agg(np.mean)
sag_perturbed_Rbp4like['Cell_id'] = sag_perturbed_Rbp4like.index.astype(str)
sag_perturbed_Rbp4like.reset_index(inplace=True,drop=True)
sag_perturbed_Rbp4like = pd.merge(sag_perturbed_Rbp4like,
                          sag_features_model.loc[:,['Cell_id','Cre_line']],on='Cell_id')
sag_model_Rbp4 = sag_features_model.loc[sag_features_model.Cre_line == "Rbp4-Cre_KL100",]
sag_perturbed_Rbp4 = pd.concat([sag_perturbed_Rbp4like,sag_model_Rbp4],sort=False).\
                    reset_index(drop=True)
sag_perturbed_Rbp4['type'] = 'Rbp4like'


sag_perturbed_Nr5like = pd.read_csv(perturbed_Nr5like_feature_file,index_col=0)
# Calculate the mean over repetitions of the perturbed models
sag_perturbed_Nr5like = sag_perturbed_Nr5like.groupby('Cell_id')[feat_list].agg(np.mean)
sag_perturbed_Nr5like['Cell_id'] = sag_perturbed_Nr5like.index.astype(str)
sag_perturbed_Nr5like.reset_index(inplace=True,drop=True)
sag_perturbed_Nr5like = pd.merge(sag_perturbed_Nr5like,
                     sag_features_model.loc[:,['Cell_id','Cre_line']],on='Cell_id')
sag_model_Nr5 = sag_features_model.loc[sag_features_model.Cre_line == "Nr5a1-Cre",]
sag_perturbed_Nr5  = pd.concat([sag_perturbed_Nr5like,sag_model_Nr5],sort=False).\
                    reset_index(drop=True)
sag_perturbed_Nr5['type'] = 'Nr5like'


sag_features_exp_selected = sag_features_exp.loc[sag_features_exp.Cell_id.isin(model_cell_ids),]
sag_features_exp_selected['type'] = 'Modeled_Exp'

sig_dict = {}
select_sag_feature ='sag_ratio1_3'
current_amp_index = select_sag_feature.split('_')[-1]
current_amp = -20*(int(current_amp_index)+1)
data_dict = {'Exp':sag_features_exp,'Modeled_Exp':sag_features_exp_selected,
             'Model':sag_features_model,
#             'Rbp4like':sag_perturbed_Rbp4,
#             'Nr5like':sag_perturbed_Nr5
             }

sig_dict_list = []
for type_,data_ in data_dict.items():

    feat_Rbp4 = data_.loc[data_.Cre_line == "Rbp4-Cre_KL100",select_sag_feature].values
    feat_Nr5 = data_.loc[data_.Cre_line == "Nr5a1-Cre",select_sag_feature].values
    _,p_val = mannwhitneyu(feat_Nr5,feat_Rbp4,alternative='two-sided')
    sig_dict_list.append({'data_type':type_,'sig_level':man_utils.pval_to_sig(p_val),
                          'Comp_type':"Nr5a1-Cre~Rbp4-Cre_KL100"})

sig_df = pd.DataFrame(sig_dict_list)
ephys_sig_group = sig_df.groupby('data_type')
sig_vars = sig_df.data_type.tolist()

sag_features_all = pd.concat([sag_features_exp,sag_features_exp_selected,
     sag_features_model,
     #sag_perturbed_Rbp4,sag_perturbed_Nr5
     ],sort=False)



palette = {cre_:cre_color_dict[cre_] for cre_ in exc_lines}


# DABEST
sag_features_all['Cre_type'] = sag_features_all.apply(lambda x:x.type+'.'+x.Cre_line,axis=1)
comp_types = sag_features_all['Cre_type'].unique().tolist()
data_types = sag_features_all.type.unique().tolist()
idx_list= []
for dtype_ in data_types:
    idx_tuple_ = ('%s.%s'%(dtype_,exc_lines[0]),'%s.%s'%(dtype_,exc_lines[1]))
    idx_list.append(idx_tuple_)

sns.set(font_scale=1)
palette_mod = {comp_type:palette[comp_type.split('.')[-1]] for comp_type in comp_types}

analysis_of_long_df = dabest.load(sag_features_all, idx=idx_list,
                                   x="Cre_type", y=select_sag_feature)

f= analysis_of_long_df.cliffs_delta.plot(custom_palette=palette_mod,
                             group_summaries='median_quartiles',swarm_desat=.9)
rawdata_axes = f.axes[0]
rawdata_axes = man_utils.annotate_sig_level(sig_vars,exc_lines,'Cre_line',
                 ephys_sig_group,'Comp_type',
                 sag_features_all,'type',select_sag_feature,rawdata_axes,
                 comp_sign='~',line_offset_factor = .05,line_height_factor=.02)

effsize_axes = f.axes[1]

raw_xticklabels = rawdata_axes.get_xticklabels()
labels = []
for label in raw_xticklabels:
    txt=label.get_text()
    type_ = txt.split('-')[0]
    num_ = txt.split('\n')[-1]
    if 'Modeled_Exp' in type_:
        type_ = type_.replace('Modeled_Exp', 'Exp $\cap$ Model')
    labels.append('%s\n%s\n%s'%(type_.split('.')[0],type_.split('.')[-1],num_))
rawdata_axes.set_xticklabels(labels)

effsize_axes.set_xticklabels(['','Rbp4-Nr5a1']*len(data_types))
rawdata_axes.set_ylabel('sag_ratio @%s$pA$'%current_amp)
f.set_size_inches(8,6)

f.savefig('figures/sag_features_comparison_dabest.svg',bbox_inches='tight')
plt.close(f)
