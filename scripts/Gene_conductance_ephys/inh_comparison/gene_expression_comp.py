import pandas as pd
import numpy as np
import os
import seaborn as sns
from ateamopt.utils import utility
import matplotlib.pyplot as plt
import feather
import man_opt

# %% Data paths
data_path = os.path.join(os.path.dirname(man_opt.__file__), os.pardir, 'assets', 'aggregated_data')
annotation_datapath = os.path.join(data_path, 'anno.feather')
me_ttype_map_path = os.path.join(data_path, 'me_ttype.pkl')
inh_expression_profile_path = os.path.join(data_path, 'inh_expression_subclass.csv')
broad_subclass_colors_filename = os.path.join(
    data_path, 'broad_subclass_colors.pkl')

# %% Read the data
annotation_df = feather.read_dataframe(annotation_datapath)
broad_subclass_color_dict = utility.load_pickle(broad_subclass_colors_filename)
inh_subclasses = ["Vip", "Sst", "Pvalb"]
inh_palette = {
    inh_subclass: broad_subclass_color_dict[inh_subclass] for inh_subclass in inh_subclasses}

inh_expression_data = pd.read_csv(inh_expression_profile_path, index_col=None)
inh_expression_data = inh_expression_data.rename(columns={'subclass_label': 'Subclass'})


# %% Filtering

# Get rid of genes expressed in less than 10% of the cells
inh_expression_data = inh_expression_data.loc[:, (inh_expression_data != 0).sum(
    axis=0) >= .1 * inh_expression_data.shape[0]]
inh_expression_data = pd.merge(inh_expression_data, annotation_df.loc[:,
                                                                      ['sample_id', 'primary_cluster_label',
                                                                       'layer_label']], on='sample_id')

# %% Plot gene expression heatmap
# Group the cells according to subclasses

sorterIndex = dict(zip(inh_subclasses, range(len(inh_subclasses))))
inh_expression_data['Subclass_rank'] = inh_expression_data['Subclass'].map(sorterIndex)

inh_expression_data = inh_expression_data.sort_values(axis=0, by='Subclass_rank')
inh_expression_data.drop('Subclass_rank', axis=1, inplace=True)

inh_genes = inh_expression_data.loc[:, ~inh_expression_data.columns.isin(['sample_id', 
                                                                          'primary_cluster_label', 'Subclass', 'layer_label'])]
expr_values = inh_genes.values.flatten()
vmax = np.nanpercentile(expr_values, 99)
tick_fontsize = 10
subclass = inh_expression_data.Subclass

row_colors = subclass.map(inh_palette)
gene_list = list(inh_genes)

subclass_genes = ['Vip', 'Sst', 'Pvalb']
subclass_numbers = {}
for subclass in inh_subclasses:
    subclass_numbers[subclass] = len(
        inh_expression_data.loc[inh_expression_data.Subclass == subclass, ])
gene_nums = inh_expression_data.shape[1]

h_genes = sorted([ch_g for ch_g in gene_list if ch_g.startswith('Hcn')])
Ca_genes = sorted([ch_g for ch_g in gene_list if ch_g.startswith('Ca')])
Na_genes = sorted([ch_g for ch_g in gene_list if ch_g.startswith('Na')])
Na_genes += sorted([ch_g for ch_g in gene_list if ch_g.startswith('Scn')])
K_genes = sorted([ch_g for ch_g in gene_list if ch_g.startswith('K')])

Kv31_gene = ['Kcnc1']
KT_genes = ['Kcnd1', 'Kcnd2', 'Kcnd3']
KP_genes = ['Kcna1', 'Kcna2', 'Kcna3', 'Kcna6']
sig_genes = Kv31_gene + KP_genes + KT_genes
sig_genes = [sig_gene for sig_gene in sig_genes if sig_gene in gene_list]
K_genes = [k_gene for k_gene in K_genes if k_gene not in sig_genes]

genes_sorted = subclass_genes + sig_genes
inh_genes = inh_genes.reindex(columns=genes_sorted)

# Generate heatmap
heatmap_figname = 'figures/expression_plot_inh.png'
utility.create_filepath(heatmap_figname)

sns.set(font_scale=.6)
g = sns.clustermap(inh_genes, vmax=vmax, row_cluster=False,
                   col_cluster=False, row_colors=row_colors, cmap='viridis',
                   cbar_kws={'label': '$\mathrm{log}_{2}(cpm+1)$'},
                   figsize=(4, 3))
ax = g.ax_heatmap
ax.set_position([.25, .2, .7, .6])
ax.plot([0, gene_nums], [subclass_numbers["Vip"], subclass_numbers["Vip"]],
        lw=1, color='k')
ax.plot([0, gene_nums], [subclass_numbers["Sst"] + subclass_numbers["Vip"],
                         subclass_numbers["Sst"] + subclass_numbers["Vip"]],
        lw=1, color='k')

ax.set_yticklabels('')
ax.tick_params(axis=u'both', which=u'both', length=0)
g.cax.set_position([.96, .2, .02, .6])
g.cax.tick_params(axis=u'both', which=u'both', length=0)

plt.setp(ax.get_xticklabels(), rotation=60)

ax_row_color = g.ax_row_colors
ax_row_color.text(.35, 20, 'Vip', rotation=90)
ax_row_color.text(.35, 350, 'Sst', rotation=90)
ax_row_color.text(.35, 750, 'Pvalb', rotation=90)
ax_row_color.set_xticklabels([''])
ax_row_color.set_position([.2, .2, .05, .6])

fig = g.fig
fig.savefig(heatmap_figname, bbox_inches='tight', dpi=400)
plt.close(fig)
