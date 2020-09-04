from umap import UMAP
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import seaborn as sns
from ateamopt.utils import utility
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
import numpy as np
import man_opt.utils as man_utils
import matplotlib.pyplot as plt
import os
from pathlib import Path

# %% Utility functions


def intLinspace(elements, numElems):
    idx = np.round(np.linspace(0, len(elements) - 1, numElems)).astype(int)
    return [elements[id_] for id_ in idx]


def create_cmap(cty):

    bcre_order = ['Htr3a', 'Sst', 'Pvalb', 'Pyr']
    pal_list = ['copper_r', 'cool_r', 'autumn_r', 'viridis_r']
    cty_order, cty_colors = [], []

    for pal, bcre_ in zip(pal_list, bcre_order):
        cre_map = [cty_ for cty_ in cty if bcre_ in cty_]
        cre_colors = sns.color_palette(pal, 5)
        cre_colors = intLinspace(cre_colors, len(cre_map))
        cty_order.extend(cre_map)
        cty_colors.extend(cre_colors)
        # if bcre_ == 'Pvalb':
        #     cty_colors.extend(cre_colors)
        # else:
        #     cty_colors.extend(['lightgrey']*len(cre_map))
    return cty_order, cty_colors


# %% Get the data
data_path = os.path.join(os.getcwd(), os.pardir,
                         os.pardir, 'assets', 'aggregated_data')
mouse_data_filename = os.path.join(data_path, 'Mouse_class_data.csv')
mouse_datatype_filename = os.path.join(data_path, 'Mouse_class_datatype.csv')
train_ephys_max_amp_filename = os.path.join(
    data_path, 'train_ephys_max_amp.csv')
train_ephys_max_amp_dtype_filename = os.path.join(
    data_path, 'train_ephys_max_amp_dtype.csv')
train_ephys_max_amp_fields_filename = os.path.join(
    data_path, 'train_ephys_max_amp_fields.json')
hof_model_ephys_max_amp_filename = os.path.join(
    data_path, 'hof_model_ephys_max_amp.csv')
hof_model_ephys_max_amp_dtype_filename = os.path.join(data_path,
                                                      'hof_model_ephys_max_amp_datatype.csv')

mouse_data_df = man_utils.read_csv_with_dtype(
    mouse_data_filename, mouse_datatype_filename)
ephys_data = man_utils.read_csv_with_dtype(train_ephys_max_amp_filename,
                                           train_ephys_max_amp_dtype_filename)
model_ephys_data = man_utils.read_csv_with_dtype(hof_model_ephys_max_amp_filename,
                                                 hof_model_ephys_max_amp_dtype_filename)

# %% Fix the cell-type label for coloring

layer_cty = 1
me_cty = 0

if layer_cty:
    cty_cluster = mouse_data_df.loc[mouse_data_df.hof_index == 0, ['Cell_id', 'Layer',
                                                                   'Broad_Cre_line', ]]
    cty_cluster.dropna(axis=0, inplace=True, how='any')
    cty_cluster['cty_layer'] = cty_cluster.apply(
        lambda x: x['Broad_Cre_line'] + '-' + 'L' + x['Layer'], axis=1)
    cty_cluster = cty_cluster.drop(columns=['Layer', 'Broad_Cre_line'])
    target_field = 'cty_layer'
    figname = Path('figures/dim_reduced_ephys_cty_layer.png')
else:
    cty_cluster = mouse_data_df.loc[mouse_data_df.hof_index == 0, [
        'Cell_id', 'me_type']]
    cty_cluster.dropna(axis=0, how='any', inplace=True)
    target_field = 'me_type'
    figname = Path('figures/dim_reduced_ephys_me.png')

# %% Prepare the data for UMAP

least_pop_index = 6
hof_num = 40
efeat_max_df = ephys_data.drop(labels=['stim_name', 'amp'], axis=1)
param_df_cty = pd.merge(efeat_max_df, cty_cluster, how='left',
                        on='Cell_id')
X_df, y_df, revised_features = man_utils.prepare_data_clf(param_df_cty, list(efeat_max_df), target_field,
                                                          least_pop=least_pop_index)

cty_order = sorted(y_df[target_field].unique().tolist())
if layer_cty:
    cty_order, cty_colors = create_cmap(cty_order)
else:
    cty_colors = sns.color_palette(palette='tab20', n_colors=len(cty_order))

cmap = ListedColormap(cty_colors)

ephys_features = [feature_ for feature_ in revised_features
                  if feature_ != 'Cell_id']

X_data = X_df.loc[:, ephys_features].values
y_df['label_encoder'] = y_df[target_field].apply(lambda x: cty_order.index(x))

dim_reduction_pipeline = Pipeline([
    ('scaling', StandardScaler()),
    ('umap', UMAP(n_neighbors=10, random_state=0))])


# %% Fit UMAP (on experimental features)
pipe_ = dim_reduction_pipeline.fit(X_data)  # Experimental ephys
data_exp = pd.concat([X_df, y_df], axis=1)
data_exp['x-dim'] = pipe_.named_steps['umap'].embedding_[:, 0]
data_exp['y-dim'] = pipe_.named_steps['umap'].embedding_[:, 1]

# Transform new data (Features of all hof models)
df_model_efeat_max = pd.merge(model_ephys_data, cty_cluster, how='left',
                              on='Cell_id')
mephys_X_df, mephys_y_df, revised_features = man_utils.prepare_data_clf(df_model_efeat_max, list(model_ephys_data),
                                                                        target_field, least_pop=hof_num * least_pop_index)
e_features = [feature_ for feature_ in revised_features if feature_ not in [
    'Cell_id', 'hof_index']]

hof_ephys_data = mephys_X_df.loc[:, e_features].values
mephys_y_df['label_encoder'] = mephys_y_df[target_field].apply(
    lambda x: cty_order.index(x))
hof_data = pd.concat([mephys_X_df, mephys_y_df], axis=1)

hof_transform = dim_reduction_pipeline.transform(hof_ephys_data)
hof_data['x-dim'] = hof_transform[:, 0]
hof_data['y-dim'] = hof_transform[:, 1]


# %% Plot the UMAP transform

title_list = ['Experiment', 'Best Model', 'Hall of Fame']
sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0, len(cty_order) - 1))
sm.set_array([])

utility.create_filepath(figname)
axis_fontsize = 10
sns.set(style='whitegrid')
fig, ax = plt.subplots(1, 3, sharey=True, figsize=(10, 4))
ax[0].scatter(data_exp['x-dim'], data_exp['y-dim'], s=10, c=data_exp['label_encoder'],
              cmap=cmap)

ax[1].scatter(hof_data.loc[hof_data.hof_index == 0, 'x-dim'],
              hof_data.loc[hof_data.hof_index == 0, 'y-dim'], s=10, c=hof_data.loc[hof_data.hof_index == 0,
                                                                                   'label_encoder'], cmap=cmap)
ax[2].scatter(hof_data['x-dim'], hof_data['y-dim'], s=2, c=hof_data['label_encoder'],
              cmap=cmap, alpha=0.6)

fig.subplots_adjust(wspace=.12)
for jj, ax_ in enumerate(ax):
    ax_.axis('off')
    ax_.set_title(title_list[jj], fontsize=axis_fontsize)
cax = fig.add_axes([0.92, 0.2, 0.01, .6])
cbar = plt.colorbar(sm, boundaries=np.arange(
    len(cty_order) + 1) - 0.5, cax=cax)
cbar.set_ticks(np.arange(len(cty_order)))
cbar.ax.set_yticklabels(cty_order, fontsize=axis_fontsize)
cbar.outline.set_visible(False)

fig.savefig(figname, bbox_inches='tight', dpi=500)
plt.close(fig)
