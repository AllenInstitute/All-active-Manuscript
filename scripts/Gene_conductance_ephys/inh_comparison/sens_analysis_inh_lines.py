import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from ateamopt.utils import utility
import os


def rename_channel(param_name_):
    param_name_ = param_name_.split('_', 1)[-1]
    param_name_ = param_name_.rsplit('_', 1)[0]
    if bool(re.search('NaT', param_name_)):
        param_name_ = 'NaT'
    elif bool(re.search('Nap', param_name_)):
        param_name_ = 'NaP'
    elif bool(re.search('K_P', param_name_)):
        param_name_ = 'KP'
    elif bool(re.search('K_T', param_name_)):
        param_name_ = 'KT'
    elif bool(re.search('Kv3_1', param_name_)):
        param_name_ = 'Kv31'
    return param_name_


filtered_me_inh_cells = utility.load_pickle('filtered_me_inh_cells.pkl')

cre_color_dict = utility.load_pickle('rnaseq_sorted_cre.pkl')
mouse_data_path = os.path.join(os.getcwd(), os.pardir, 'Mouse_class_data.csv')
mouse_datatype_path = os.path.join(
    os.getcwd(), os.pardir, 'Mouse_class_datatype.csv')

datatypes = pd.read_csv(mouse_datatype_path)['types']
mouse_data = pd.read_csv(mouse_data_path, dtype=datatypes.to_dict())
mouse_data = mouse_data.loc[mouse_data.hof_index == 0, ]
cre_cluster = mouse_data.loc[mouse_data.hof_index ==
                             0, ['Cell_id', 'Cre_line']]
sens_analysis_datapath = '/allen/aibs/mat/anin/Sobol_analysis/Inh_cells/*/'\
    'sensitivity_data/*.csv'
sens_analysis_pathlist = glob.glob(sens_analysis_datapath)
cell_id_list = [path_.split('/')[-3] for path_ in sens_analysis_pathlist]
sa_df_inh_lines = [pd.read_csv(path_, index_col=None) for path_ in
                   sens_analysis_pathlist]
for ii, cell_id in enumerate(cell_id_list):
    sa_df_inh_lines[ii]['Cell_id'] = cell_id
sa_df_inh_lines = pd.concat(sa_df_inh_lines, sort=False, ignore_index=True)
sa_df_inh_lines = pd.merge(sa_df_inh_lines, cre_cluster, on='Cell_id',
                           how='left')

select_features = ['AP_width', 'AHP_depth', 'Spikecount']
sa_df_inh_lines = sa_df_inh_lines.loc[(sa_df_inh_lines.feature.isin(select_features)) &
                                      (sa_df_inh_lines.Cell_id.isin(filtered_me_inh_cells)), ]
sa_df_inh_lines['param_name'] = sa_df_inh_lines['param_name'].apply(
    lambda x: rename_channel(x))
channel_sorted = sorted(
    sa_df_inh_lines.param_name.unique().tolist(), reverse=True)

inh_lines = ['Htr3a-Cre_NO152', 'Sst-IRES-Cre', 'Pvalb-IRES-Cre']
palette = {inh_line_: cre_color_dict[inh_line_] for inh_line_ in inh_lines}
tick_fontsize = 10
axis_fontsize = 10
g = sns.FacetGrid(data=sa_df_inh_lines, col='feature',
                  col_order=select_features)
g = g.map(sns.barplot, 'param_name', 'sobol_index', 'Cre_line',
          order=channel_sorted, hue_order=inh_lines,
          palette=palette,
          errwidth=1,
          linewidth=0)

axes = g.axes.ravel()

for ax in axes:
    plt.setp(ax.get_xticklabels(), rotation=60, ha='right',
             fontsize=12)
    title_ = ax.get_title().split('=')[-1]
    title_ = title_.replace('Spikecount', 'spike frequency')
    ax.set_title(title_, fontsize=axis_fontsize)
    ax.set_xlabel('')
    ax.grid(False)
    ax.set_ylabel(ax.get_ylabel(), fontsize=axis_fontsize)
    plt.setp(ax.get_xticklabels(), fontsize=tick_fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=tick_fontsize)

h, l = ax.get_legend_handles_labels()
fig = g.fig
fig.legend(handles=h, labels=l, loc='lower center', ncol=3,
           fontsize=axis_fontsize, frameon=False)
fig.tight_layout(rect=[0, .05, 1, .95])
#fig.subplots_adjust(wspace = .15)
fig.set_size_inches((8, 4))
fig.savefig('Sens_analysis_inh.pdf', bbox_inches='tight')
plt.close(fig)
