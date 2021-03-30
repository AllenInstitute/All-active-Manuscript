import pandas as pd
import os
import man_opt.utils as man_utils
import man_opt
from man_opt.statistical_tests import pairwise_comp
import dabest
import seaborn as sns
import matplotlib.pyplot as plt
import feather
from ateamopt.utils import utility

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
layer_cty_colors_filename = os.path.join(data_path, 'layer_cty_colors.pkl')

# %% Read the data

mouse_data = man_utils.read_csv_with_dtype(
    mouse_data_filename, mouse_datatype_filename)
bcre_cluster = mouse_data.loc[mouse_data.hof_index ==
                              0, ['Cell_id', 'Broad_Cre_line', 'Layer', 'Dendrite_type']]
hof_param_data = man_utils.read_csv_with_dtype(
    param_data_filename, param_datatype_filename)
hof_param_data = hof_param_data.loc[hof_param_data.hof_index == 0, ]
hof_param_data_cre = pd.merge(
    hof_param_data, bcre_cluster, how='left', on='Cell_id')

param_pyr = hof_param_data_cre.loc[hof_param_data_cre.Broad_Cre_line == 'Pyr', ]

sup_pyr = param_pyr.loc[(param_pyr.Layer == '2/3') &
                        (param_pyr.Dendrite_type == 'spiny'), 'Cell_id'].tolist()

deep_pyr = param_pyr.loc[(param_pyr.Layer == '5') &
                         (param_pyr.Dendrite_type == 'spiny'), 'Cell_id'].tolist()

filtered_pyr_cells = sup_pyr + deep_pyr
param_pyr = param_pyr.loc[param_pyr.Cell_id.isin(filtered_pyr_cells), ]
param_pyr['depth'] = param_pyr['Layer'].apply(
    lambda x: 'L2/3 PC' if x == '2/3' else 'L5 PC')
depth_cty = param_pyr.depth.unique().tolist()
all_cre_lines = mouse_data.loc[mouse_data.Cell_id.isin(
    filtered_pyr_cells), 'Cre_line'].unique().tolist()

# Save cells list
cell_list_path = os.path.join(data_path, 'pyr_cells_list.csv')
pyr_cell_list = param_pyr.loc[:, ['Cell_id', 'Layer', 'depth']]
pyr_cell_list.to_csv(cell_list_path, index=False)

layer_cty_colors = utility.load_pickle(layer_cty_colors_filename)

# %% Statistical comparison

conductance_params = [dens_ for dens_ in list(
    param_pyr) if dens_.startswith('gbar')]
param_pyr = param_pyr.dropna(axis=1, how='all')
param_pyr = param_pyr.dropna(subset=[cond for cond in list(param_pyr)
                                     if cond in conductance_params])

select_conds = ['gbar_Ih.apical', 'gbar_Ih.somatic',
                'gbar_NaTa_t.axonal',
                'gbar_Kv3_1.axonal', 'gbar_Kv3_1.somatic',
                'gbar_K_Tst.axonal', 'gbar_K_Tst.somatic',
                'gbar_Ca_LVA.axonal', 'gbar_Ca_LVA.somatic']

diff_param_df = pairwise_comp(
    param_pyr, 'depth', depth_cty, select_conds)

diff_param_df['param'] = diff_param_df['param'].apply(lambda x:
                                                      man_utils.replace_channel_name(x) + '.' + x.split('.')[-1])
cond_sig_grouped = diff_param_df.groupby('param')

# %% Visualize estimation statistics

param_pyr_select = param_pyr.loc[:, select_conds + ['depth']]
param_data = pd.melt(param_pyr_select, id_vars='depth', var_name='conductance',
                     value_name='value')
param_data['conductance'] = param_data['conductance'].apply(lambda x:
                                                            man_utils.replace_channel_name(x) + '.' + x.split('.')[-1])

cond_types = param_data.conductance.unique().tolist()
depth_types = sorted(param_data.depth.unique().tolist(), reverse=True)
param_data['param_depth'] = param_data.apply(lambda x: x.conductance + '.' +
                                             x.depth, axis=1)

palette_channel = {type_: layer_cty_colors['Pyr-L5'] if type_.split('.')[-1] == 'L5 PC'
                   else layer_cty_colors['Pyr-L2/3']
                   for type_ in param_data.param_depth.unique()}


# %% Gene expression comparison

annotation_datapath = os.path.join(data_path, 'anno.feather')
expression_profile_path = os.path.join(data_path, 'l23_vs_l5_expression.csv')

annotation_df = feather.read_dataframe(annotation_datapath)
expression_data = pd.read_csv(expression_profile_path, index_col=0)
expression_data = expression_data.rename(columns={'cre_label': 'Cre_line'})
expression_data = expression_data.loc[:, (expression_data != 0).sum(
    axis=0) >= .1 * expression_data.shape[0]]
expression_data = pd.merge(expression_data, annotation_df.loc[:,
                                                              ['sample_id', 'primary_cluster_label',
                                                               'class_label', 'layer_label']],
                           on='sample_id')

expression_pyr = expression_data.loc[expression_data.class_label ==
                                     'Glutamatergic', ]

sup_pyr = expression_data.loc[(expression_data.layer_label == 'L2/3-L4'),
                              'sample_id'].tolist()

deep_pyr = expression_data.loc[expression_data.layer_label.isin(['L5', 'L5-L6']),
                               'sample_id'].tolist()

filtered_pyr_cells = sup_pyr + deep_pyr
expression_pyr = expression_pyr.loc[expression_pyr.sample_id.isin(
    filtered_pyr_cells), ]
expression_pyr['depth'] = expression_pyr['layer_label'].apply(
    lambda x: 'L2/3 PC' if x == 'L2/3-L4' else 'L5 PC')

# %% Statistical comparison


select_genes = ['Hcn1', 'Hcn2', 'Scn8a', 'Kcnc1', 'Kcnd1',
                'Kcnd2', 'Kcnd3', 'Kcnab1', 'Kcnip1', 'Kcnip2', 'Cacna1g']
expressed_genes = [
    select_gene for select_gene in select_genes if select_gene in list(expression_pyr)]
expression_pyr = expression_pyr.dropna(axis=1, how='all')
expression_pyr = expression_pyr.dropna(subset=[gene_ for gene_ in list(expression_pyr)
                                               if gene_ in expressed_genes])

diff_gene_df = pairwise_comp(
    expression_pyr, 'depth', depth_cty, expressed_genes)
gene_sig_grouped = diff_gene_df.groupby('param')

# %% Visualize estimation statistics

gene_channel_dict = {
    'Ih': ['Hcn1'],
    # 'NaT': ['Scn8a'],
    # 'KT': ['Kcnd1', 'Kcnd2', 'Kcnd3', 'Kcnab1', 'Kcnip1', 'Kcnip2'],
    # 'Kv31': ['Kcnc1'],
    # 'CaLV': ['Cacna1g']
}

expression_select = expression_pyr.loc[:, expressed_genes + ['depth']]
gene_expr = pd.melt(expression_select, id_vars='depth', var_name='genes',
                    value_name='value')
gene_types = gene_expr.genes.unique().tolist()
depth_types = sorted(gene_expr.depth.unique().tolist())
gene_expr['gene_depth'] = gene_expr.apply(lambda x: x.genes + '.' +
                                          x.depth, axis=1)
palette_gene = {type_: layer_cty_colors['Pyr-L5'] if type_.split('.')[-1] == 'L5 PC' else
                layer_cty_colors['Pyr-L2/3']
                for type_ in gene_expr.gene_depth.unique()}

sns.set(font_scale=1.5)
for channel, genes in gene_channel_dict.items():
    fig, ax = plt.subplots(ncols=2, figsize=(15, 7),
                           gridspec_kw={"hspace": 0.5,
                                        "width_ratios": [1.75, 1]})

    channel_select = [
        channel_ for channel_ in cond_types if channel in channel_]
    idx_channel = []
    for dtype_ in channel_select:
        idx_tuple_ = ('%s.%s' %
                      (dtype_, depth_types[0]), '%s.%s' % (dtype_, depth_types[1]))
        idx_channel.append(idx_tuple_)

    param_data_select = param_data.loc[param_data.conductance.isin(
        channel_select), ]

    analysis_df_channel = dabest.load(param_data_select, idx=idx_channel,
                                      x='param_depth', y='value')

    analysis_df_channel.cliffs_delta.plot(ax=ax[0], custom_palette=palette_channel,
                                          group_summaries='median_quartiles', swarm_desat=.9)

    ax[0] = man_utils.annotate_sig_level(channel_select, depth_types, 'depth',
                                         cond_sig_grouped, 'Comp_type', param_data_select,
                                         'conductance', 'value', ax[0])

    ax[0].ticklabel_format(style='sci', scilimits=(0, 1), axis='y')
    ax[0].set_ylabel(r'$\mathrm{\bar{g}\;(S\:cm^{-2})}$')
    raw_xticklabels = ax[0].get_xticklabels()
    labels = []
    for label in raw_xticklabels:
        txt = label.get_text()
        type_ = txt.split('-')[0]
        param_ = type_.rsplit('.', 1)[0]
        cty_ = type_.split('.')[-1].split('\n')[0]
        num_ = txt.split('\n')[-1]
        labels.append('%s\n%s\n%s' % (param_, cty_, num_))
    ax[0].set_xticklabels(labels)

#    ax[0].set_title(r'-log(p-val) = %.2f' % -np.log10(cond_p_val))

    genes = [gene for gene in genes if gene in gene_types]
    gene_expr_correlate = gene_expr.loc[gene_expr.genes.isin(genes), ]

    idx_gene = []
    for gene in genes:
        idx_tuple_ = ('%s.%s' %
                      (gene, depth_types[0]), '%s.%s' % (gene, depth_types[1]))
        idx_gene.append(idx_tuple_)

    analysis_df_gene = dabest.load(
        gene_expr_correlate, idx=idx_gene, x='gene_depth', y='value')

    analysis_df_gene.cliffs_delta.plot(ax=ax[1], custom_palette=palette_gene,
                                       group_summaries='median_quartiles', swarm_desat=.9,
                                       float_contrast=True)
    ax[1] = man_utils.annotate_sig_level(genes, depth_types, 'depth',
                                         gene_sig_grouped, 'Comp_type', gene_expr_correlate,
                                         'genes', 'value', ax[1])
    ax[1].set_ylabel(r'$\mathrm{log}_{2}(cpm+1)$')
    raw_xticklabels = ax[1].get_xticklabels()
    labels = []
    for label in raw_xticklabels:
        txt = label.get_text()
        type_ = txt.split('-')[0]
        param_ = type_.rsplit('.', 1)[0]
        cty_ = type_.split('.')[-1].split('\n')[0]
        num_ = txt.split('\n')[-1]
        labels.append('%s\n%s\n%s' % (param_, cty_, num_))
    ax[1].set_xticklabels(labels)

    fig.savefig(os.path.join('figures/%s_l23vl5.svg' % channel), bbox_inches='tight')
    plt.close(fig)

# %%
