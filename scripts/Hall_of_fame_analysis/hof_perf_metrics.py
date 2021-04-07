import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from ateamopt.utils import utility
import man_opt.utils as man_utils
import feather

tick_fontsize = 12
axis_fontsize = 14


def get_metrics_df(model_perf_filelist):
    metric_list_df = []
    for metric_file_ in model_perf_filelist:
        cell_id = metric_file_.split('/')[-2]
        metric_list = utility.load_pickle(metric_file_)
        metric_dict = [{'hof_index': ii, 'Feature_Avg_Train': metric_list_['Feature_Average'],
                        'Feature_Avg_Generalization': metric_list_['Feature_Average_Generalization'],
                        'Explained_Variance': metric_list_['Explained_Variance'],
                        'Seed_Index': metric_list_['Seed'],
                        'Cell_id': cell_id}
                       for ii, metric_list_ in enumerate(metric_list)]
        metric_list_df.extend(metric_dict)
    perf_metric_df = pd.DataFrame(metric_list_df)
    perf_metric_df['Explained_Variance'] *= 100
    return perf_metric_df


def get_data_fields(path_or_data):
    if isinstance(path_or_data, pd.DataFrame):
        return list(path_or_data)
    else:
        if path_or_data.endswith('.json'):
            json_data = utility.load_json(path_or_data)
            return list(json_data.keys())
        elif path_or_data.endswith('.csv'):
            csv_data = pd.read_csv(path_or_data, index_col=None)
            return list(csv_data)

# %%  Read in data


data_path = os.path.join(os.getcwd(), os.pardir,
                         os.pardir, 'assets', 'aggregated_data')
mouse_data_filename = os.path.join(data_path, 'Mouse_class_data.csv')
mouse_datatype_filename = os.path.join(data_path, 'Mouse_class_datatype.csv')
model_performance_metric_path = '/allen/aibs/mat/ateam_shared/' \
    'Mouse_Model_Fit_Metrics/*/exp_variance_hof.pkl'
broad_subclass_colors_filename = os.path.join(
    data_path, 'broad_subclass_colors.pkl')
me_ttype_map_path = os.path.join(data_path, 'me_ttype.pkl')
annotation_datapath = os.path.join(data_path, 'anno.feather')

annotation_df = feather.read_dataframe(annotation_datapath)
subclass_colors = annotation_df.loc[:, ['subclass_label', 'subclass_color']]
subclass_colors = subclass_colors.drop_duplicates().set_index('subclass_label')[
    'subclass_color'].to_dict()
broad_subclass_color_dict = utility.load_pickle(broad_subclass_colors_filename)

mouse_data_df = man_utils.read_csv_with_dtype(
    mouse_data_filename, mouse_datatype_filename)
mouse_data_df = man_utils.add_transcriptomic_subclass(
    mouse_data_df, me_ttype_map_path)
mouse_data_df = man_utils.add_broad_subclass(mouse_data_df)
ttype_cluster = mouse_data_df.loc[mouse_data_df.
                                  hof_index == 0, ['Cell_id', 'ttype']]
broad_subclass_cluster = mouse_data_df.loc[mouse_data_df.hof_index == 0, [
    'Cell_id', 'Broad_subclass']]

model_perf_filelist = glob.glob(model_performance_metric_path)
model_perf_data = get_metrics_df(model_perf_filelist)
perf_fields = get_data_fields(model_perf_data)


metrics = ['Feature_Avg_Train', 'Feature_Avg_Generalization', 'Explained_Variance']
id_vars = [field_ for field_ in list(model_perf_data) if field_ not in metrics]


# %% Hall of Fame Performance

model_perf_data_melt = model_perf_data.melt(id_vars=id_vars, value_vars=metrics,
                                            var_name='perf_metric', value_name='metric_val')
sns.set(style='whitegrid')
g = sns.FacetGrid(col='perf_metric', data=model_perf_data_melt,
                  sharey=False, col_order=metrics)
g.map(sns.boxplot, 'hof_index', 'metric_val', showfliers=False, linewidth=1,
      color='#5481a6cc')
axes = g.axes.ravel()
for ii, ax in enumerate(axes):
    ax.set_xticks(np.arange(0, 40, 10))
    ax.set_xticklabels([str(hof_) for hof_ in np.arange(0, 40, 10)])
    ax.set_title(metrics[ii], fontsize=tick_fontsize)
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), fontsize=tick_fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=tick_fontsize)
    ax.set_xlabel('hof_index', fontsize=tick_fontsize)
    if ii == 0:
        ax.set_ylabel('z-score')
    elif ii == len(metrics)-1:
        ax.set_ylabel('%', labelpad=-5, fontsize=tick_fontsize)

g.fig.subplots_adjust(wspace=.25)
g.fig.set_size_inches(12, 4)
g.fig.savefig('figures/eval_metrics_with_hof.pdf', bbox_inches='tight')
plt.close(g.fig)

# %% Best model performance across different cell-types

mouse_perf_meta = perf_fields + ['ttype', 'Broad_subclass',
                                 'Dendrite_type']
mouse_perf_best = mouse_data_df.loc[mouse_data_df.hof_index == 0,
                                    mouse_perf_meta]

subclass_group = mouse_perf_best.groupby('ttype')
subclass_group = subclass_group['Cell_id'].agg([np.size])
survived_subclasses = subclass_group.loc[subclass_group['size'] >= 5, ].index.tolist()
subclass_order = ["Lamp5", "Sncg", "Vip", "Sst", "Pvalb", "L2/3 IT", "L4",
                  "L5 IT", "L5 PT", "NP", "L6 IT", "L6 CT", "L6b"]
broad_subclass_order = ['Vip', 'Sst', 'Pvalb', 'Pyr', 'Other']

# Training Error and Explained Variance

for metric_, metric_unit in zip(['Feature_Avg_Train', 'Explained_Variance'], ['z-score', '%']):
    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(12, 4),
                           gridspec_kw={'width_ratios': [1, 1.5, 4]})

    perf_comparison_fields = ['Dendrite_type', 'Broad_subclass', 'ttype']
    for i, comp_field_ in enumerate(perf_comparison_fields):
        if comp_field_ == 'ttype':
            order = [subclass for subclass in subclass_order if subclass in survived_subclasses]
            color_list = [subclass_colors[ind] for ind in order]
            select_df = mouse_perf_best.loc[mouse_perf_best[comp_field_].isin(
                order), ]

        elif comp_field_ == 'Broad_subclass':
            select_df = mouse_perf_best.loc[mouse_perf_best[comp_field_].isin(
                broad_subclass_order), ]
            order = broad_subclass_order
            color_list = list(broad_subclass_color_dict.values())
        else:
            select_df = mouse_perf_best
            order = sorted(select_df[comp_field_].unique().tolist())
            color_list = ['steelblue'] * len(order)

        ax[i] = sns.barplot(x=comp_field_, y=metric_, data=select_df, order=order,
                            palette=color_list, alpha=.8, ax=ax[i], errwidth=1, linewidth=0)

        ax[i].set_xticklabels(order, rotation=60, ha='right', fontsize=tick_fontsize)
        ax[i].grid(False)
        if i == 0:
            ax[i].set_ylabel('%s (%s)' % (metric_, metric_unit),
                             labelpad=10, fontsize=axis_fontsize)
        else:
            ax[i].set_ylabel('')
        ax[i].set_xlabel('')
        sns.despine(ax=ax[i])

    plt.setp(ax[0].get_yticklabels(), fontsize=tick_fontsize)
    fig.savefig('figures/%s.pdf' % metric_, bbox_inches='tight')
    plt.close(fig)

# Generalization Error
fig, ax = plt.subplots(1, 3, sharey=True, figsize=(12, 6),
                       gridspec_kw={'width_ratios': [1, 1.5, 4]})

perf_comparison_fields = ['Dendrite_type', 'Broad_subclass', 'ttype']
for i, comp_field_ in enumerate(perf_comparison_fields):
    perf_group = mouse_perf_best.groupby(comp_field_)
    perf_group_agg = perf_group['Feature_Avg_Generalization'].agg([np.mean,
                                                                   np.std, np.size])
    perf_group_agg['std_err'] = perf_group_agg['std']\
        / np.sqrt(perf_group_agg['size'])
    perf_group_agg = perf_group_agg.loc[perf_group_agg['size'] >= 5, ]
    perf_group_agg = perf_group_agg.sort_values(comp_field_)
    color_list = ['steelblue'] * len(perf_group_agg.index)

    ax[i].bar(perf_group_agg.index, perf_group_agg['mean'],
              yerr=perf_group_agg['std_err'], error_kw={'elinewidth': 1},
              color=color_list)
    ax[i].set_xticklabels(perf_group_agg.index, rotation=90,
                          ha='center', fontsize=tick_fontsize)
    ax[i].grid(axis='x')
    sns.despine(ax=ax[i])

plt.setp(ax[0].get_yticklabels(), fontsize=tick_fontsize)
ax[0].set_ylabel('Generalization error (z-score)', labelpad=15,
                 fontsize=axis_fontsize)
fig.tight_layout()
fig.savefig('figures/Gen_err_comp_cty.pdf', bbox_inches='tight')
plt.close(fig)
