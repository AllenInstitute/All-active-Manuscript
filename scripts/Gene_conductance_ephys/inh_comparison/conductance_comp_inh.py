import pandas as pd
import numpy as np
import os
from scipy.stats import mannwhitneyu
import seaborn as sns
from collections import defaultdict
import man_opt.utils as man_utils
import man_opt
import matplotlib.pyplot as plt
from ateamopt.utils import utility
from itertools import combinations
from statsmodels.stats.multitest import fdrcorrection


# %% Data paths

data_path = os.path.join(os.path.dirname(
    man_opt.__file__), os.pardir, 'assets', 'aggregated_data')
cre_coloring_filename = os.path.join(data_path, 'cre_color_tasic16.pkl')
inh_expression_profile_path = os.path.join(data_path, 'inh_expression_all.csv')

mouse_data_filename = os.path.join(data_path, 'Mouse_class_data.csv')
mouse_datatype_filename = os.path.join(data_path, 'Mouse_class_datatype.csv')

param_data_filename = os.path.join(data_path, 'allactive_params.csv')
param_datatype_filename = os.path.join(
    data_path, 'allactive_params_datatype.csv')
me_cluster_data_filename = os.path.join(data_path, 'tsne_mouse.csv')

filtered_inh_expression_path = os.path.join(
    data_path, 'inh_expression_filtered.csv')
# %% Read the data

mouse_data = man_utils.read_csv_with_dtype(
    mouse_data_filename, mouse_datatype_filename)
cre_cluster = mouse_data.loc[mouse_data.hof_index ==
                             0, ['Cell_id', 'Cre_line']]
hof_param_data = man_utils.read_csv_with_dtype(
    param_data_filename, param_datatype_filename)
hof_param_data = hof_param_data.loc[hof_param_data.hof_index == 0, ]
hof_param_data_cre = pd.merge(
    hof_param_data, cre_cluster, how='left', on='Cell_id')
model_cre_lines = sorted(cre_cluster.Cre_line.unique().tolist())
me_cluster_data = pd.read_csv(me_cluster_data_filename, index_col=0)
me_cluster_data.rename(columns={'specimen_id': 'Cell_id'}, inplace=True)
me_cluster_data['Cell_id'] = me_cluster_data['Cell_id'].astype(str)
cre_color_dict = utility.load_pickle(cre_coloring_filename)

inh_lines = ["Htr3a-Cre_NO152", "Sst-IRES-Cre", "Pvalb-IRES-Cre"]

hof_param_data_cre = pd.merge(
    hof_param_data_cre, me_cluster_data, on='Cell_id')
hof_param_data_inh_lines = hof_param_data_cre.loc[hof_param_data_cre.Cre_line.isin(
    inh_lines), ]
pv_basket_me = hof_param_data_inh_lines.loc[(me_cluster_data.morph != 'CHC') &
                                            (hof_param_data_inh_lines.Cre_line == 'Pvalb-IRES-Cre') &
                                            (hof_param_data_inh_lines.dendrite_type == 'aspiny'), 'Cell_id'].tolist()

sst_cells_me = hof_param_data_inh_lines.loc[(hof_param_data_inh_lines.Cre_line == 'Sst-IRES-Cre') &
                                            (hof_param_data_inh_lines.dendrite_type == 'aspiny'), 'Cell_id'].tolist()

htr3a_cells_me = hof_param_data_inh_lines.loc[(hof_param_data_inh_lines.Cre_line == 'Htr3a-Cre_NO152') & (hof_param_data_inh_lines.dendrite_type == 'aspiny'),
                                              'Cell_id'].tolist()

filtered_me_inh_cells = pv_basket_me+sst_cells_me+htr3a_cells_me
utility.save_pickle(os.path.join(
    data_path, 'filtered_me_inh_cells.pkl'), filtered_me_inh_cells)
palette = {inh_line_: cre_color_dict[inh_line_] for inh_line_ in inh_lines}

hof_param_data_inh_lines = hof_param_data_inh_lines.loc[hof_param_data_inh_lines.Cell_id.isin(
    filtered_me_inh_cells)]

# %% Pairwise comparison between inh lines

conductance_params = [dens_ for dens_ in list(hof_param_data_inh_lines) if dens_ not in [
    'Cell_id', 'hof_index', 'Broad_Cre_line'] and dens_.startswith('gbar')]

somatic_params = [
    cond_param for cond_param in conductance_params if 'somatic' in cond_param]

sig_level = .05
num_comparisons = len(inh_lines)**2 - len(inh_lines)


diff_channel_cond_df = []
p_val_list = []

# One sided mann-whitney u test
for param_ in somatic_params:
    for comb in combinations(inh_lines, 2):
        cre_x, cre_y = comb
        param_x = hof_param_data_inh_lines.loc[hof_param_data_inh_lines.Cre_line ==
                                               cre_x, param_].values
        param_y = hof_param_data_inh_lines.loc[hof_param_data_inh_lines.Cre_line ==
                                               cre_y, param_].values
        _, p_val_x = mannwhitneyu(param_x, param_y, alternative='less')
        _, p_val_y = mannwhitneyu(param_y, param_x, alternative='less')
        comp_type = '%s<%s' % (
            cre_x, cre_y) if p_val_x < p_val_y else '%s<%s' % (cre_y, cre_x)
        p_val = min(p_val_x, p_val_y)
        sig_dict = {'Comp_type': comp_type,
                    'param': param_}
        diff_channel_cond_df.append(sig_dict)
        p_val_list.append(p_val)

# FDR correction @5%
_, p_val_corrected = fdrcorrection(p_val_list)


diff_channel_cond_df = pd.DataFrame(diff_channel_cond_df)
diff_channel_cond_df['p_val'] = p_val_corrected
diff_channel_cond_df['sig_level'] = diff_channel_cond_df['p_val'].apply(
    lambda x: man_utils.pval_to_sig(x))
diff_channel_cond_df = diff_channel_cond_df.loc[diff_channel_cond_df.sig_level != 'n.s.', ]

significant_parameters = sorted(diff_channel_cond_df.param.unique().tolist())
conductance_sig_grouped = diff_channel_cond_df.groupby('param')

inh_expression_data = pd.read_csv(filtered_inh_expression_path, index_col=None)
channel_correlate_dict = defaultdict(list)

gene_types = [col_ for col_ in list(inh_expression_data) if col_ not in ['sample_id', 'Cre_line',
                                                                         'primary_cluster_label', 'subclass_label', 'layer_label']]
gene_types = sorted(gene_types)
Kv31_gene = ['Kcnc1']
KT_genes = ['Kcnd1', 'Kcnd2', 'Kcnd3']
KP_genes = ['Kcna1', 'Kcna2', 'Kcna3', 'Kcna6']

for gene_ in gene_types:
    if gene_ in Kv31_gene:
        channel_correlate_dict['Kv3_1'].append(gene_)
    elif gene_ in KP_genes:
        channel_correlate_dict['K_Pst'].append(gene_)
    elif gene_ in KT_genes:
        channel_correlate_dict['K_Tst'].append(gene_)

diff_gene_expression_df = []
p_val_list = []
for gene_ in gene_types:
    for comb in combinations(inh_lines, 2):
        cre_x, cre_y = comb
        gene_x = inh_expression_data.loc[inh_expression_data.Cre_line ==
                                         cre_x, gene_].values
        gene_y = inh_expression_data.loc[inh_expression_data.Cre_line ==
                                         cre_y, gene_].values
        _, p_val_x = mannwhitneyu(gene_x, gene_y, alternative='less')
        _, p_val_y = mannwhitneyu(gene_y, gene_x, alternative='less')
        comp_type = '%s<%s' % (
            cre_x, cre_y) if p_val_x < p_val_y else '%s<%s' % (cre_y, cre_x)
        p_val = min(p_val_x, p_val_y)
        sig_dict = {'Comp_type': comp_type,
                    'gene': gene_, }
        diff_gene_expression_df.append(sig_dict)
        p_val_list.append(p_val)

_, p_val_corrected = fdrcorrection(p_val_list)


diff_gene_expression_df = pd.DataFrame(diff_gene_expression_df)
diff_gene_expression_df['p_val'] = p_val_corrected
diff_gene_expression_df['sig_level'] = diff_gene_expression_df['p_val'].apply(
    lambda x: man_utils.pval_to_sig(x))
diff_gene_expression_df = diff_gene_expression_df.loc[diff_gene_expression_df.sig_level != 'n.s.', ]
gene_sig_grouped = diff_gene_expression_df.groupby('gene')


diff_param_data_inh = pd.melt(hof_param_data_inh_lines, id_vars=['Cell_id', 'Cre_line'],
                              value_vars=significant_parameters, var_name='conductance',
                              value_name='value')


inh_expr_df = pd.melt(inh_expression_data, id_vars=['sample_id', 'Cre_line'],
                      value_vars=gene_types, var_name='gene', value_name='cpm')
hue_levels = inh_lines

tick_fontsize = 16
axis_fontsize = 16
sns.set(style='whitegrid')

for channel_, genes in channel_correlate_dict.items():
    cond_ = 'gbar_%s.somatic' % channel_
    if cond_ not in significant_parameters:
        continue
    diff_param_data_inh_ = diff_param_data_inh.loc[diff_param_data_inh.conductance == cond_, ]
    inh_expr_df_ = inh_expr_df.loc[inh_expr_df.gene.isin(genes), ]
    inh_expr_df_['cpm'] = inh_expr_df_['cpm'].astype(float)
    hue_var = 'Cre_line'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), sharey=False,
                                   gridspec_kw={'width_ratios': [1, 1]})
    ax1 = sns.boxplot(x='conductance', y='value', data=diff_param_data_inh_,
                        order=[cond_],
                        hue_order=hue_levels, hue=hue_var, showfliers=False,
                        palette=palette, ax=ax1)
    ax1 = man_utils.annotate_sig_level([cond_], hue_levels, hue_var,
                                       conductance_sig_grouped, 'Comp_type', diff_param_data_inh_, 'conductance', 'value',
                                       ax1, plot_type='seaborn', line_height_factor=.5)
    ax1.ticklabel_format(style='sci', scilimits=(0, 2), axis='y')
    ticklabels = [man_utils.replace_channel_name(channel.split('.')[0].split('_', 1)[1]) for channel in
                  [cond_]]
    ax1.set_xticklabels(ticklabels, fontsize=tick_fontsize)
    plt.setp(ax1.get_yticklabels(), fontsize=tick_fontsize)
    ax1.set_ylabel(r'$\mathrm{\bar{g}\:(S cm^{-2})}$',
                   fontsize=axis_fontsize+2)
    ax1.set_xlabel('channel', fontsize=axis_fontsize)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height]
                     )  # resize position
    ax1.legend(loc='lower center', bbox_to_anchor=(1, - 0.2),
               ncol=3, fontsize=axis_fontsize, frameon=False)
    ax1.set_yscale("log")
    hue_var = 'Cre_line'
    sns.despine(ax=ax1)

    ax2 = sns.boxplot(x='gene', y='cpm', data=inh_expr_df_, hue=hue_var,
                      order=sorted(genes), hue_order=hue_levels, showfliers=False,
                      palette=palette, ax=ax2)
    ax2 = man_utils.annotate_sig_level(sorted(genes), hue_levels, hue_var,
                                       gene_sig_grouped, 'Comp_type', inh_expr_df_, 'gene', 'cpm',
                                       ax2, plot_type='seaborn')
    ax2.ticklabel_format(style='sci', scilimits=(0, 2), axis='y')
    ax2.set_ylabel(r'$\mathrm{log}_{2}(cpm+1)$', fontsize=axis_fontsize + 2)
    ax2.set_xlabel(ax2.get_xlabel(), fontsize=axis_fontsize)
    plt.setp(ax2.get_xticklabels(), fontsize=tick_fontsize)
    plt.setp(ax2.get_yticklabels(), fontsize=tick_fontsize)
    ax2.get_legend().remove()

    sns.despine(ax=ax2)
    for ax in [ax1, ax2]:
        ax.set_xlabel('')
        ax.grid(False)

    fig.subplots_adjust(wspace=.35)
    figname = 'figures/diff_expression_%s_inh.pdf' % channel_
    utility.create_filepath(figname)
    fig.savefig(figname, bbox_inches='tight')

    plt.close(fig)
