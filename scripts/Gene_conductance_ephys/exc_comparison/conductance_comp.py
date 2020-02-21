import pandas as pd
import numpy as np
import os
import seaborn as sns
from ateamopt.utils import utility
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import man_opt
import man_opt.utils as man_utils
import dabest
from statsmodels.stats.multitest import fdrcorrection
from itertools import combinations
from scipy.stats import mannwhitneyu

# Data paths
data_path = os.path.join(os.path.dirname(man_opt.__file__),os.pardir,'assets','aggregated_data')
mouse_data_filename = os.path.join(data_path,'Mouse_class_data.csv')
mouse_datatype_filename = os.path.join(data_path,'Mouse_class_datatype.csv')
hof_param_data_filename = os.path.join(data_path,'allactive_params.csv')
hof_param_datatypes_filename = os.path.join(data_path,'allactive_params_datatype.csv')
cre_coloring_filename = os.path.join(data_path,'rnaseq_sorted_cre.pkl')
sdk_data_filename = os.path.join(data_path,'sdk.csv')
sdk_datatype_filename = os.path.join(data_path,'sdk_datatype.csv')

# Read the data
hof_num = 1 # Only work with the best model
hof_param_data = man_utils.read_csv_with_dtype(hof_param_data_filename, hof_param_datatypes_filename)
best_param_data = hof_param_data.loc[hof_param_data.hof_index < hof_num,hof_param_data.columns != 'hof_index']
cre_color_dict = utility.load_pickle(cre_coloring_filename)

mouse_data_df = man_utils.read_csv_with_dtype(mouse_data_filename,mouse_datatype_filename)
mouse_data_best = mouse_data_df.loc[mouse_data_df.hof_index == 0,]
mouse_metadata = mouse_data_best.loc[mouse_data_best.hof_index == 0, ['Cell_id', 'Cre_line','Layer']]
best_param_with_metadata =  pd.merge(best_param_data,mouse_metadata, how='left', on='Cell_id')
sdk_data= man_utils.read_csv_with_dtype(sdk_data_filename,sdk_datatype_filename)


exc_lines = ["Nr5a1-Cre","Rbp4-Cre_KL100"]
exc_palette = {exc_line_:cre_color_dict[exc_line_] for exc_line_ in exc_lines}

select_params_somatic = ['gbar_Ih.somatic']
best_param_exc_lines = best_param_with_metadata.loc[best_param_with_metadata.Cre_line.isin(exc_lines),]

# Filtering on the ME side
Nr5_filtered = sdk_data.loc[(sdk_data.structure__layer == '4') &
                        (sdk_data.line_name=='Nr5a1-Cre'),'Cell_id'].tolist()
Rbp4_filtered =   sdk_data.loc[(sdk_data.structure__layer == '5') &
                        (sdk_data.line_name=='Rbp4-Cre_KL100'),'Cell_id'].tolist()
filtered_exc_cells = Nr5_filtered + Rbp4_filtered
utility.save_pickle(os.path.join(data_path,'filtered_me_exc_cells.pkl'),filtered_exc_cells)

best_param_exc_lines = best_param_exc_lines.loc[best_param_exc_lines.Cell_id.isin(filtered_exc_cells),]

diff_cond_somatic = []
p_val_list = []

# Pairwise comparison of Ih conductance density
for param_ in select_params_somatic:
    for comb in combinations(exc_lines, 2):
        cre_x,cre_y = comb
        param_x = best_param_exc_lines.loc[best_param_exc_lines.Cre_line == cre_x,param_].values
        param_y = best_param_exc_lines.loc[best_param_exc_lines.Cre_line == cre_y,param_].values
        _,p_val_x= mannwhitneyu(param_x,param_y,alternative='less')
        _,p_val_y = mannwhitneyu(param_y,param_x,alternative='less')
        comp_type = '%s<%s'%(cre_x,cre_y) if p_val_x<p_val_y else '%s<%s'%(cre_y,cre_x)
        p_val = min(p_val_x,p_val_y)
        sig_dict = {'Comp_type' : comp_type,
                    'param': param_}
        diff_cond_somatic.append(sig_dict)
        p_val_list.append(p_val)

diff_cond_somatic = pd.DataFrame(diff_cond_somatic)
diff_cond_somatic['p_val'] = p_val_list
diff_cond_somatic['sig_level'] = diff_cond_somatic['p_val'].apply(lambda x:man_utils.pval_to_sig(x))
sig_params_somatic = sorted(diff_cond_somatic.param.unique().tolist())
cond_sig_grouped = diff_cond_somatic.groupby('param')

best_param_melt_df = pd.melt(best_param_exc_lines,id_vars=['Cell_id','Cre_line'],
                  value_vars=select_params_somatic,var_name='conductance',
                  value_name='value')
best_param_melt_df = best_param_melt_df.loc[best_param_melt_df.conductance == select_params_somatic[0],]
best_param_melt_df['cond_cre'] = best_param_melt_df.apply(lambda x:x.conductance+'.'+x.Cre_line,
                        axis=1)
short_param = select_params_somatic[0].split('.')[0].replace('gbar_','')


comp_types = best_param_melt_df['cond_cre'].unique().tolist()
palette_mod = {comp_type:exc_palette[comp_type.split('.')[-1]] for comp_type in comp_types}
idx_tuple = ('%s.%s'%(select_params_somatic[0],exc_lines[0]),
                 '%s.%s'%(select_params_somatic[0],exc_lines[1]))


# Plot Ih conductance density comparison
sns.set(font_scale=1)
Ih_analysis_aa_df = dabest.load(best_param_melt_df, idx=idx_tuple,
                                   x="cond_cre", y='value')

f= Ih_analysis_aa_df.cliffs_delta.plot(custom_palette=palette_mod,
                             group_summaries='median_quartiles',swarm_desat=.85,
                             swarm_ylim=(1e-7,1.3*1e-4))
rawdata_axes = f.axes[0]
rawdata_axes.ticklabel_format(style='sci',scilimits=(0,1),axis='y')
rawdata_axes.set_ylabel(r'$\mathrm{\bar{g}\;(S\:cm^{-2})}$')
rawdata_axes.yaxis.set_major_locator(ticker.MaxNLocator(2))
effsize_axes = f.axes[1]
effsize_axes.yaxis.set_major_locator(ticker.MaxNLocator(4))

raw_xticklabels = rawdata_axes.get_xticklabels()
labels = []
for label in raw_xticklabels:
    txt=label.get_text()
    type_ = txt.split('-')[0]
    cre_= type_.split('.')[-1]
    num_ = txt.split('\n')[-1]
    labels.append('%s\n%s\n%s'%(short_param,cre_,num_))
rawdata_axes.set_xticklabels(labels)
rawdata_axes = man_utils.annotate_sig_level(select_params_somatic,exc_lines,'Cre_line',
                 cond_sig_grouped,'Comp_type',best_param_melt_df,'conductance','value',rawdata_axes)

effsize_axes.set_xticklabels(['','Rbp4-Nr5a1'])
f.subplots_adjust(hspace=.3)
f.set_size_inches(3,4)
f.savefig('figures/conductance_comp_exc.svg',bbox_inches='tight')
plt.close(f)
