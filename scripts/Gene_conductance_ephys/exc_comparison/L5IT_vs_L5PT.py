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


param_pyr = hof_param_data_ttype.loc[hof_param_data_ttype.ttype.isin(
    ['L2/3 IT', 'L5 PT', 'L5 IT'])]

exc_subclasses = ["L5 IT", "L5 PT"]
l5_PT = param_pyr.loc[(param_pyr.ttype == 'L5 PT') &
                      (param_pyr.Dendrite_type == 'spiny'), 'Cell_id'].tolist()

l5_IT = param_pyr.loc[(param_pyr.ttype == 'L5 IT') &
                      (param_pyr.Dendrite_type == 'spiny'), 'Cell_id'].tolist()

filtered_me_cells = l5_PT + l5_IT
param_pyr = param_pyr.loc[param_pyr.Cell_id.isin(filtered_me_cells), ]

annotation_df = feather.read_dataframe(annotation_datapath)
tasic_colors = annotation_df.loc[:, ['subclass_label', 'subclass_color']]
tasic_colors = tasic_colors.drop_duplicates().set_index('subclass_label')[
    'subclass_color'].to_dict()

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

# %% Gene expression comparison

expression_data = pd.read_csv(expression_profile_path, index_col=0)
expression_data = expression_data.rename(columns={'cre_label': 'Cre_line'})
expression_data = expression_data.loc[:, (expression_data != 0).sum(
    axis=0) >= .1 * expression_data.shape[0]]

# %% Visualize estimation statistics

sns.set(font_scale=1.2)
param_pyr_select = param_pyr.loc[:, select_conds + ['Cell_id', 'ttype']]
param_data = pd.melt(param_pyr_select, id_vars=['Cell_id', 'ttype'], var_name='conductance',
                     value_name='value')
param_data['conductance'] = param_data['conductance'].apply(lambda x:
                                                            man_utils.replace_channel_name(x) + '.' + x.split('.')[-1])

cond_types = param_data.conductance.unique().tolist()
param_data['param_ttype'] = param_data.apply(lambda x: x.conductance + '.' +
                                             x.ttype, axis=1)

palette_channel = {type_: tasic_colors[type_.split('.')[-1]] for type_ in param_data.param_ttype.unique()}


# %% Gene expression comparison

expression_data = pd.read_csv(expression_profile_path, index_col=0)
expression_data = expression_data.rename(columns={'cre_label': 'Cre_line'})
expression_data = expression_data.loc[:, (expression_data != 0).sum(
    axis=0) >= .1 * expression_data.shape[0]]
expression_data = pd.merge(expression_data, annotation_df.loc[:,
                                                              ['sample_id', 'primary_cluster_label',
                                                               'class_label', 'layer_label']],
                           on='sample_id')

l5_PT_trans = expression_data.loc[(expression_data.subclass_label == 'L5 PT'),
                                  'sample_id'].tolist()

l5_IT_trans = expression_data.loc[(expression_data.subclass_label == 'L5 IT'),
                                  'sample_id'].tolist()

filtered_t_cells = l5_PT_trans + l5_IT_trans
trans_expression = expression_data.loc[expression_data.sample_id.isin(
    filtered_t_cells), ]

# %% Statistical comparison

gene_channel_dict = {
    'Ih': ['Hcn1', 'Hcn2', 'Hcn3'],
    'NaP': ['Scn1a', 'Scn3a', 'Scn8a'],
    'NaT': ['Scn1a', 'Scn3a', 'Scn8a'],
    'KP': ['Kcna1', 'Kcna2', 'Kcna3', 'Kcna4', 'Kcna5', 'Kcna6',
           'Kcna7', 'Kcna10'],
    'KT': ['Kcnd1', 'Kcnd2', 'Kcnd3', 'Kcnab1', 'Kcnip1', 'Kcnip2'],
    'Kv31': ['Kcnc1']
}

select_genes = []
for channel, genes in gene_channel_dict.items():
    select_genes.extend(genes)

expressed_genes = [
    select_gene for select_gene in select_genes if select_gene in list(trans_expression)]
trans_expression = trans_expression.dropna(axis=1, how='all')
trans_expression = trans_expression.dropna(subset=[gene_ for gene_ in list(trans_expression)
                                                   if gene_ in expressed_genes])

diff_gene_df = pairwise_comp(
    trans_expression, 'subclass_label', exc_subclasses, expressed_genes)
gene_sig_grouped = diff_gene_df.groupby('param')

# %% Visualize estimation statistics


expression_select = trans_expression.loc[:,
                                         expressed_genes + ['subclass_label']]
gene_expr = pd.melt(expression_select, id_vars='subclass_label', var_name='genes',
                    value_name='value')
gene_types = gene_expr.genes.unique().tolist()
gene_expr['gene_ttype'] = gene_expr.apply(lambda x: x.genes + '.' +
                                          x.subclass_label, axis=1)
palette_gene = {type_: tasic_colors[type_.split(
    '.')[-1]] for type_ in gene_expr.gene_ttype.unique()}

for channel, genes in gene_channel_dict.items():

    fig, ax = plt.subplots(ncols=2, figsize=(15, 6),
                           gridspec_kw={"wspace": 0.3,
                                        "hspace": 0.3,
                                        "width_ratios": [1, 1.5]}, dpi=100)

    channel_select = [
        channel_ for channel_ in cond_types if channel in channel_]
    idx_channel = []
    for dtype_ in channel_select:
        idx_tuple_ = ('%s.%s' %
                      (dtype_, exc_subclasses[0]), '%s.%s' % (dtype_, exc_subclasses[1]))
        idx_channel.append(idx_tuple_)

    param_data_select = param_data.loc[param_data.conductance.isin(
        channel_select), ]

    analysis_df_channel = dabest.load(param_data_select, idx=idx_channel,
                                      x='param_ttype', y='value')

    analysis_df_channel.cliffs_delta.plot(ax=ax[0], custom_palette=palette_channel,
                                          group_summaries='median_quartiles', swarm_desat=.9)

    ax[0] = man_utils.annotate_sig_level(channel_select, exc_subclasses, 'ttype',
                                         cond_sig_grouped, 'Comp_type', param_data_select, 'conductance', 'value', ax[0])

#    ax[0].set_title(r'-log(p-val) = %.2f' % -np.log10(cond_p_val))

    genes = [gene for gene in genes if gene in gene_types]
    gene_expr_correlate = gene_expr.loc[gene_expr.genes.isin(genes), ]

    idx_gene = []
    for gene in genes:
        idx_tuple_ = ('%s.%s' %
                      (gene, exc_subclasses[0]), '%s.%s' % (gene, exc_subclasses[1]))
        idx_gene.append(idx_tuple_)

    analysis_df_gene = dabest.load(
        gene_expr_correlate, idx=idx_gene, x='gene_ttype', y='value')

    analysis_df_gene.cliffs_delta.plot(ax=ax[1], custom_palette=palette_gene,
                                       group_summaries='median_quartiles', swarm_desat=.9,
                                       float_contrast=True)
    ax[1] = man_utils.annotate_sig_level(genes, exc_subclasses, 'subclass_label',
                                         gene_sig_grouped, 'Comp_type', gene_expr_correlate, 'genes', 'value', ax[1])

    plt.show()
