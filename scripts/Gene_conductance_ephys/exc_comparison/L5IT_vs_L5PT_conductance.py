import pandas as pd
import os
import man_opt.utils as man_utils
import man_opt
from man_opt.statistical_tests import pairwise_comp
import seaborn as sns
import matplotlib.pyplot as plt
import feather
from ateamopt.utils import utility


# %% Data paths

data_path = os.path.join(os.path.dirname(
    man_opt.__file__), os.pardir, 'assets', 'aggregated_data')
inh_expression_profile_path = os.path.join(data_path, 'inh_expression_all.csv')
mouse_data_filename = os.path.join(data_path, 'Mouse_class_data.csv')
mouse_datatype_filename = os.path.join(data_path, 'Mouse_class_datatype.csv')

param_data_filename = os.path.join(data_path, 'allactive_params.csv')
param_datatype_filename = os.path.join(
    data_path, 'allactive_params_datatype.csv')
me_ttype_map_path = os.path.join(data_path, 'me_ttype.pkl')
annotation_datapath = os.path.join(data_path, 'anno.feather')
expression_profile_path = os.path.join(data_path, 'exc_ttype_expression.csv')

# %% Read the data

mouse_data = man_utils.read_csv_with_dtype(
    mouse_data_filename, mouse_datatype_filename)
me_ttype_map = utility.load_pickle(me_ttype_map_path)

metype_cluster = mouse_data.loc[mouse_data.hof_index == 0, [
    'Cell_id', 'Dendrite_type', 'me_type']]
hof_param_data = man_utils.read_csv_with_dtype(
    param_data_filename, param_datatype_filename)
hof_param_data = hof_param_data.loc[hof_param_data.hof_index == 0, ]
hof_param_data_ttype = pd.merge(
    hof_param_data, metype_cluster, how='left', on='Cell_id').dropna(how='any', subset=['me_type'])

hof_param_data_ttype['ttype'] = hof_param_data_ttype['me_type'].apply(
    lambda x: me_ttype_map[x])

exc_subclasses = ["L5 IT", "L5 PT"]
param_pyr = hof_param_data_ttype.loc[hof_param_data_ttype.ttype.isin(exc_subclasses)]
l5_PT = param_pyr.loc[(param_pyr.ttype == 'L5 PT') &
                      (param_pyr.Dendrite_type == 'spiny'), 'Cell_id'].tolist()

l5_IT = param_pyr.loc[(param_pyr.ttype == 'L5 IT') &
                      (param_pyr.Dendrite_type == 'spiny'), 'Cell_id'].tolist()

filtered_me_cells = l5_PT + l5_IT
param_pyr = param_pyr.loc[param_pyr.Cell_id.isin(filtered_me_cells), ]

annotation_df = feather.read_dataframe(annotation_datapath)
subclass_colors = annotation_df.loc[:, ['subclass_label', 'subclass_color']]
subclass_colors = subclass_colors.drop_duplicates().set_index('subclass_label')[
    'subclass_color'].to_dict()
palette = {exc_subclass: subclass_colors[exc_subclass]
           for exc_subclass in exc_subclasses}

# %% Statistical comparison

conductance_params = [dens_ for dens_ in list(
    param_pyr) if dens_.startswith('gbar')]
param_pyr = param_pyr.dropna(axis=1, how='all')
param_pyr = param_pyr.dropna(subset=[cond for cond in list(param_pyr)
                                     if cond in conductance_params])

select_conds = [
    'gbar_Ih.apical', 'gbar_Ih.somatic',
    'gbar_NaTa_t.axonal', 'gbar_NaTs2_t.somatic',
    'gbar_Nap_Et2.axonal', 'gbar_Nap_Et2.somatic',
    'gbar_Kv3_1.axonal', 'gbar_Kv3_1.somatic',
    'gbar_K_Tst.axonal', 'gbar_K_Tst.somatic',
    'gbar_K_Pst.axonal', 'gbar_K_Pst.somatic'
]

diff_param_df = pairwise_comp(
    param_pyr, 'ttype', exc_subclasses, select_conds)

diff_param_df['param'] = diff_param_df['param'].apply(lambda x:
                                                      man_utils.replace_channel_name(x) + '.' + x.split('.')[-1])
cond_sig_grouped = diff_param_df.groupby('param')

param_select = ['gbar_Ih.somatic', 'gbar_Ih.apical', 'gbar_NaTa_t.axonal']
id_vars = ['Cell_id', 'ttype']
select_cond_df = param_pyr.loc[:, id_vars + param_select]
select_param_melt = pd.melt(select_cond_df, id_vars=id_vars, value_vars=param_select,
                            var_name='cond')
select_param_melt['cond'] = select_param_melt['cond'].apply(
    lambda x: man_utils.replace_channel_name(x.split('.')[0]) + '.' + x.split('.')[-1])
param_select_renamed = ['Ih.somatic', 'Ih.apical', 'NaT.axonal']

# %% Box plot for each conductance

sns.set(style='whitegrid')

fig, ax = plt.subplots(1, len(param_select), sharey=False, figsize=(5, 3))
for ii, cond_ in enumerate(param_select_renamed):
    data = select_param_melt.loc[select_param_melt.cond == cond_, ]
    sig_df = diff_param_df.loc[diff_param_df.param == cond_, :]
    sig_vars = sig_df.param.tolist()
    ephys_sig_group = sig_df.groupby('param')
    ax[ii] = sns.boxplot(x='cond', y='value', data=data, hue='ttype',
                         hue_order=exc_subclasses, ax=ax[ii], palette=palette, linewidth=1, showfliers=False)
    for patch in ax[ii].artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .7))
    ax[ii] = sns.stripplot(x='cond', y='value', data=data, hue='ttype',
                           hue_order=exc_subclasses, ax=ax[ii], palette=palette,
                           alpha=0.8, dodge=True)

    ax[ii] = man_utils.annotate_sig_level(sig_vars, exc_subclasses, 'ttype',
                                          ephys_sig_group, 'Comp_type',
                                          data, 'cond', 'value', ax[ii], 
                                          plot_type='normal')
    ax[ii].grid(False)
    sns.despine(ax=ax[ii])
    ax[ii].get_legend().remove()
    ax[ii].set_xlabel('')
    ax[ii].set_xticklabels([cond_])
    ax[ii].set_ylabel('')
    ax[ii].ticklabel_format(axis='y', style='sci', scilimits=[1e-8, 1e-2])
    ax[ii].yaxis.major.formatter._useMathText = True

fig.subplots_adjust(wspace=.7)
fig.savefig('figures/L5IT_vs_L5PT_cond.svg', bbox_inches='tight')
plt.close(fig)
