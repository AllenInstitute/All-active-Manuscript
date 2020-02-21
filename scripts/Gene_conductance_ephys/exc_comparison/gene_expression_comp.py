import pandas as pd
import numpy as np
import os
import seaborn as sns
from ateamopt.utils import utility
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import feather
import man_opt
import man_opt.utils as man_utils
import dabest
from statsmodels.stats.multitest import fdrcorrection
from itertools import combinations
from scipy.stats import mannwhitneyu


#%% Data paths
data_path = os.path.join(os.path.dirname(man_opt.__file__),os.pardir,'assets','aggregated_data')
annotation_datapath = os.path.join(data_path,'anno.feather')
cre_coloring_filename = os.path.join(data_path,'rnaseq_sorted_cre.pkl')
exc_expression_profile_path = os.path.join(data_path,'exc_expression_all.csv')

#%% Read the data

annotation_df = feather.read_dataframe(annotation_datapath)
cre_color_dict = utility.load_pickle(cre_coloring_filename)
exc_lines = ["Nr5a1-Cre","Rbp4-Cre_KL100"]
exc_palette = {exc_line_:cre_color_dict[exc_line_] for exc_line_ in exc_lines}

exc_expression_data = pd.read_csv(exc_expression_profile_path,index_col=None)
exc_expression_data = exc_expression_data.rename(columns={'cre_label':'Cre_line'})


#%% Filtering

# Get rid of genes expressed in less than 10% of the cells
exc_expression_data = exc_expression_data.loc[:,(exc_expression_data != 0).sum(axis=0) >= .1* exc_expression_data.shape[0]]
exc_expression_data = pd.merge(exc_expression_data, annotation_df.loc[:,
                  ['sample_id','primary_cluster_label','subclass_label','layer_label']],
                    on='sample_id')
h_genes = sorted([ch_g for ch_g in list(exc_expression_data) if ch_g.startswith('Hcn')])
HCN_expression_data = exc_expression_data.loc[:,h_genes]

# Filtering on Transciptomic side
Nr5_filtered = exc_expression_data.loc[(exc_expression_data.primary_cluster_label.\
        str.contains('L4(.*)Rspo1')) & (exc_expression_data.Cre_line=='Nr5a1-Cre') &
        (exc_expression_data.layer_label == 'L4'),
                'sample_id'].tolist()

Rbp4_filtered =  exc_expression_data.loc[(~exc_expression_data.primary_cluster_label.\
          str.contains('L4(.*)Rspo1')) & (exc_expression_data.layer_label == 'L5') &
            (~exc_expression_data.primary_cluster_label.str.contains('L5(.*)Hsd11b1'))&
            (exc_expression_data.Cre_line=='Rbp4-Cre_KL100') & 
            (~exc_expression_data.subclass_label.isin(['Sst','Vip'])),
                'sample_id'].tolist()

filtered_exc_cells = Nr5_filtered + Rbp4_filtered
utility.save_pickle(os.path.join(data_path,'filtered_ttype_exc_cells.pkl'),filtered_exc_cells)
exc_expression_data = exc_expression_data.loc[exc_expression_data.sample_id.isin(filtered_exc_cells),]

#%% Plot gene expression heatmap
# Group the cells according to cre-lines
sorterIndex = dict(zip(exc_lines,range(len(exc_lines))))
exc_expression_data['Cre_rank'] = exc_expression_data['Cre_line']. map(sorterIndex)
exc_expression_data = exc_expression_data.sort_values(axis=0,by='Cre_rank')
exc_expression_data.drop('Cre_rank',axis= 1,inplace=True)

exc_genes = exc_expression_data.loc[:,~exc_expression_data.columns.isin(['sample_id','Cre_line',
              'primary_cluster_label','subclass_label','layer_label'])]
expr_values = exc_genes.values.flatten()
vmax = np.nanpercentile(expr_values,99)


tick_fontsize = 10
cre = exc_expression_data.Cre_line

row_colors = cre.map(exc_palette)
gene_list = list(exc_genes)

Cre_genes = ['Nr5a1','Rbp4']

cre_numbers = {}
for cre in exc_lines:
    cre_numbers[cre] = len(exc_expression_data.loc[exc_expression_data.Cre_line == cre,])
gene_nums = exc_expression_data.shape[1]

# Looking at marker and h-current genes
genes_sorted = Cre_genes + h_genes
exc_genes = exc_genes.reindex(columns=genes_sorted)
genes_sorted = Cre_genes + h_genes
exc_genes = exc_genes.reindex(columns=genes_sorted)

# Generate heatmap
heatmap_figname = 'figures/expression_plot_exc.png'
utility.create_filepath(heatmap_figname)

sns.set(font_scale=.6)
g = sns.clustermap(exc_genes,vmax=vmax,row_cluster=False,
               col_cluster=False,row_colors=row_colors,cmap='coolwarm',
               cbar_kws={'label':'$\mathrm{log}_{2}(cpm+1)$'},
               figsize=(3,2))
ax = g.ax_heatmap
ax.set_position([.25, .2, .7, .6])
ax.plot([0,gene_nums],[cre_numbers['Nr5a1-Cre'],cre_numbers['Nr5a1-Cre']],
        lw=1,color='k')
ax.set_yticklabels('')
ax.set_yticks([])

g.cax.set_position([.97, .2, .02, .6])
#g.cax.yaxis.set_major_locator(ticker.FixedLocator([0,6*1e-4]))
g.cax.tick_params(axis=u'both', which=u'both',length=0)

ax_row_color = g.ax_row_colors
ax_row_color.text(.2,25,'Nr5a1',rotation=90)
ax_row_color.text(.2,500,'Rbp4',rotation=90)
ax_row_color.set_xticklabels([''])
ax_row_color.set_xticklabels([''])
ax_row_color.set_position([.2, .2, .05, .6])
fig = g.fig
fig.savefig(heatmap_figname,bbox_inches='tight',dpi=300)
plt.close(fig)

#%% Pairwise comparison between HCN expressions for Nr5a1 vs Rbp4
diff_gene_expression_df = []
p_val_list = []

# One sided mann-whitney u test
for gene_ in h_genes:
    for comb in combinations(exc_lines, 2):
        cre_x,cre_y = comb
        gene_x = exc_expression_data.loc[exc_expression_data.Cre_line == cre_x,gene_].values
        gene_y = exc_expression_data.loc[exc_expression_data.Cre_line == cre_y,gene_].values
        _,p_val_x = mannwhitneyu(gene_x,gene_y,alternative='less')
        _,p_val_y = mannwhitneyu(gene_y,gene_x,alternative='less')
        comp_type = '%s<%s'%(cre_x,cre_y) if p_val_x<p_val_y else '%s<%s'%(cre_y,cre_x)
        p_val = min(p_val_x,p_val_y)
        sig_dict = {'Comp_type' : comp_type,
            'gene': gene_,}
        diff_gene_expression_df.append(sig_dict)
        p_val_list.append(p_val)

# FDR correction @5%
_,p_val_corrected = fdrcorrection(p_val_list)

diff_gene_expression_df = pd.DataFrame(diff_gene_expression_df)
diff_gene_expression_df['p_val'] = p_val_corrected
diff_gene_expression_df['sig_level'] = diff_gene_expression_df['p_val'].apply(
                                lambda x:man_utils.pval_to_sig(x))
diff_gene_expression_df = diff_gene_expression_df.loc[diff_gene_expression_df.sig_level != 'n.s.',]
gene_sig_grouped = diff_gene_expression_df.groupby('gene')


exc_expression_melted = pd.melt(exc_expression_data,id_vars=['sample_id','Cre_line'],
                  value_vars=h_genes,var_name='gene',value_name='cpm')

exc_expression_melted['Cre_gene'] = exc_expression_melted.apply(lambda x:x.gene+'.'+
                     x.Cre_line,axis=1)
comp_types = exc_expression_melted['Cre_gene'].unique().tolist()
data_types = exc_expression_melted.gene.unique().tolist()

idx_list= []
for dtype_ in data_types:
    idx_tuple_ = ('%s.%s'%(dtype_,exc_lines[0]),'%s.%s'%(dtype_,exc_lines[1]))
    idx_list.append(idx_tuple_)

#%% Cummings plot

palette_mod = {comp_type:exc_palette[comp_type.split('.')[-1]] for comp_type in comp_types}
sns.set(font_scale=1.2)
gene_comp_figname = 'figures/diff_gene_expression_exc.svg'
gene_df = dabest.load(exc_expression_melted, idx=idx_list,x="Cre_gene", y='cpm')
f= gene_df.cliffs_delta.plot(custom_palette=palette_mod,
                 group_summaries='median_quartiles',swarm_desat=.9,
#                 swarm_ylim=(1e-5,1e-3),
                 swarmplot_kwargs={'size':2})

rawdata_axes = f.axes[0]
rawdata_axes = man_utils.annotate_sig_level(data_types,exc_lines,'Cre_line',
                 gene_sig_grouped,'Comp_type',
                 exc_expression_melted,'gene','cpm',rawdata_axes,line_offset_factor=.25)


effsize_axes = f.axes[1]
raw_xticklabels = rawdata_axes.get_xticklabels()
labels = []

for label in raw_xticklabels:
    txt=label.get_text()
    type_ = txt.split('-')[0]
    num_ = txt.split('\n')[-1]
    labels.append('%s\n%s\n%s'%(type_.split('.')[0],type_.split('.')[-1],num_))
rawdata_axes.set_xticklabels(labels)

effsize_axes.set_xticklabels(['','Rbp4-Nr5a1']*len(data_types))
rawdata_axes.set_ylabel(r'$\mathrm{log}_{2}(cpm+1)$')
f.savefig(gene_comp_figname,bbox_inches='tight')
f.set_size_inches(4,4)
plt.close(f)
