import pandas as pd
import os
from ateamopt.utils import utility
import seaborn as sns
import matplotlib.pyplot as plt
import man_opt.utils as man_utils
import dabest
import man_opt
import feather


def get_feature_vec(feature_dict_list, select_features, cell_id, cre):
    feature_vec_dict = {}
    for ii, feat_dict in enumerate(feature_dict_list):
        for feature in select_features:
            try:
                feature_vec_dict['%s_%s' % (feature, ii)] = feat_dict[feature][0]
            except:
                feature_vec_dict['%s_%s' % (feature, ii)] = None

    feature_vec_dict['Cell_id'] = cell_id
    feature_vec_dict['Cre_line'] = cre
    return feature_vec_dict

# %% Data paths


data_path = os.path.join(os.path.dirname(
    man_opt.__file__), os.pardir, 'assets', 'aggregated_data')
mouse_data_filename = os.path.join(data_path, 'Mouse_class_data.csv')
mouse_datatype_filename = os.path.join(data_path, 'Mouse_class_datatype.csv')
me_ttype_map_path = os.path.join(data_path, 'me_ttype.pkl')

sdk_data_filename = os.path.join(data_path, 'sdk.csv')
sdk_datatype_filename = os.path.join(data_path, 'sdk_datatype.csv')
annotation_datapath = os.path.join(data_path, 'anno.feather')

mouse_data = man_utils.read_csv_with_dtype(
    mouse_data_filename, mouse_datatype_filename)
me_ttype_map = utility.load_pickle(me_ttype_map_path)

metype_cluster = mouse_data.loc[mouse_data.hof_index == 0, [
    'Cell_id', 'Dendrite_type', 'me_type']]

sdk_data = man_utils.read_csv_with_dtype(
    sdk_data_filename, sdk_datatype_filename)
sdk_data = sdk_data.rename(columns={'line_name': 'Cre_line',
                                    'ef__threshold_i_long_square': 'rheobase',
                                    'ef__f_i_curve_slope': 'fi_slope'})
sdk_data = pd.merge(
    sdk_data, metype_cluster, how='left', on='Cell_id').dropna(how='any', subset=['me_type'])

sdk_data['ttype'] = sdk_data['me_type'].apply(
    lambda x: me_ttype_map[x])
exc_ttypes = ["L5 CF", "L5 IT"]

sdk_data = sdk_data.loc[sdk_data.ttype.isin(exc_ttypes), ['Cell_id', 'me_type', 'ttype',
                                                          'Cre_line', 'rheobase', 'fi_slope']]

cell_ids = list(set(sdk_data.Cell_id.tolist()))
efel_feature_path = 'eFEL_features_ttype'

# %% Aggregate spiking and subthreshold features

spiking_features_df_list = []
subthresh_features_df_list = []

spiking_features = ['AP_amplitude_from_voltagebase', 'AP1_amp', 'APlast_amp',
                    "AP_width", 'AHP_depth', 'mean_frequency', 'Spikecount', 'inv_first_ISI', 'voltage_base']
subthresh_features = ['sag_amplitude', 'sag_ratio1', 'sag_ratio2']

num_supthresh_stims = 4
num_subthresh_stims = 4

metadata_list = ['Cell_id', 'Cre_line']


for idx, cell_id in enumerate(cell_ids):
    cell_efeatures_dir = os.path.join(efel_feature_path, cell_id)
    cell_protocols_filename = os.path.join(cell_efeatures_dir, 'protocols.json')
    cell_features_filename = os.path.join(efel_feature_path, cell_id, 'features.json')
    stimmap_data_path = os.path.join(efel_feature_path, cell_id, 'StimMapReps.csv')

    stimmap_data = pd.read_csv(stimmap_data_path, sep='\s*,\s*',
                               header=0, encoding='ascii', engine='python')
    stimmap_data['Amplitude_Start'] *= 1e12
    stimmap_data['Amplitude_Start'] = round(stimmap_data['Amplitude_Start'], 2)
    stimmap_data['Amplitude_End'] *= 1e12

    cell_features = utility.load_json(cell_features_filename)
    cell_protocols = utility.load_json(cell_protocols_filename)

    stimmap_data = stimmap_data.sort_values(by='Amplitude_Start')
    cre = sdk_data.loc[sdk_data.Cell_id == cell_id, 'Cre_line'].values[0]
    rheobase = sdk_data.loc[sdk_data.Cell_id == cell_id, 'rheobase'].values[0]

    spiking_stims = stimmap_data.loc[stimmap_data['Amplitude_Start'] >= rheobase,
                                     'DistinctID'].tolist()
    hyperpolarizing_stims = list(reversed(stimmap_data.loc[stimmap_data['Amplitude_Start'] < 0,
                                                           'DistinctID'].tolist()))

    spiking_feature_dicts = [cell_features[stim_name]['soma'] for stim_name in spiking_stims
                             if stim_name in cell_features.keys()]
    subthresh_feature_dicts = [cell_features[stim_name]['soma'] for stim_name in hyperpolarizing_stims
                               if stim_name in cell_features.keys()]
    spiking_features_df_list.append(get_feature_vec(
        spiking_feature_dicts, spiking_features, cell_id, cre))
    subthresh_features_df_list.append(get_feature_vec(
        subthresh_feature_dicts, subthresh_features, cell_id, cre))


spiking_features_df = pd.DataFrame(spiking_features_df_list)
subthresh_features_df = pd.DataFrame(subthresh_features_df_list)

indexed_spiking_features = [
    feat + '_' + str(ii) for ii in range(num_supthresh_stims) for feat in spiking_features]
indexed_subthresh_features = [
    feat + '_' + str(ii) for ii in range(num_subthresh_stims) for feat in subthresh_features]

spiking_features_df = spiking_features_df.iloc[:, spiking_features_df.columns.get_indexer(
    indexed_spiking_features + metadata_list).tolist()]
subthresh_features_df = subthresh_features_df.iloc[:, subthresh_features_df.columns.get_indexer(
    indexed_subthresh_features + ['Cell_id']).tolist()]
features_df = pd.merge(spiking_features_df, subthresh_features_df, on='Cell_id', )

# Compare features between t-types
features_df = pd.merge(features_df, sdk_data.loc[:, [
    'Cell_id', 'fi_slope', 'ttype']], on='Cell_id')

stim_index = 3  # 60pA above rheobase
select_features = ['AP_width', 'AHP_depth',
                   'AP_amplitude_from_voltagebase', 'sag_ratio1']

select_features = [feat_ + '_%s' % stim_index for feat_ in select_features]

rename_dict = {'AP_width_3': 'AP_width', 'AHP_depth_3': 'AHP_depth',
               'AP_amplitude_from_voltagebase_3': 'AP_amplitude',
               'sag_ratio1_3': 'sag_ratio'}
select_features_df = features_df.loc[:, select_features + ['fi_slope', 'Cell_id', 'ttype']]
select_features_df = features_df.rename(columns=rename_dict)

feature_select = list(rename_dict.values()) + ['fi_slope']
unit_list = ['$ms$', '$mV$', '$Hz$', '$Hz/pA$']

select_ttypes = ["L5 CF", "L5 IT"]
diff_ephys_df = man_opt.statistical_tests.pairwise_comp(select_features_df, 'ttype',
                                                        select_ttypes, feature_select)
diff_ephys_df = diff_ephys_df.rename(columns={'param': 'feature'})
feat_sig_grouped = diff_ephys_df.groupby('feature')


# %% Estimation statistics - DABEST

feature_data = pd.melt(select_features_df, id_vars=['Cell_id', 'ttype'],
                       value_vars=feature_select, var_name='features', value_name='value')

feature_data['feat_ttype'] = feature_data.apply(lambda x: x.features + '.' +
                                                x.ttype, axis=1)

annotation_df = feather.read_dataframe(annotation_datapath)
tasic_colors = annotation_df.loc[:, ['subclass_label', 'subclass_color']]
tasic_colors = tasic_colors.drop_duplicates().set_index('subclass_label')[
    'subclass_color'].to_dict()

palette_features = {type_: tasic_colors[type_.split('.')[-1]] if type_.split('.')[-1] != 'L5 CF' else tasic_colors['L5 PT']
                    for type_ in feature_data.feat_ttype.unique()}


sns.set(font_scale=1.2)
fig, ax = plt.subplots(figsize=(15, 8))
idx_feat = []
for dtype_ in feature_select:
    idx_tuple_ = ('%s.%s' %
                  (dtype_, select_ttypes[0]), '%s.%s' % (dtype_, select_ttypes[1]))
    idx_feat.append(idx_tuple_)

feature_data_select = feature_data.loc[feature_data.features.isin(feature_select), ]
analysis_df_feat = dabest.load(feature_data_select, idx=idx_feat,
                               x='feat_ttype', y='value')

f = analysis_df_feat.cliffs_delta.plot(ax=ax, custom_palette=palette_features,
                                       group_summaries='median_quartiles', swarm_desat=.9)
ax = man_utils.annotate_sig_level(feature_select, select_ttypes, 'ttype',
                                  feat_sig_grouped, 'Comp_type', feature_data_select, 'features', 'value', ax)

rawdata_axes = f.axes[0]
raw_xticklabels = rawdata_axes.get_xticklabels()
labels = []
for label in raw_xticklabels:
    txt = label.get_text()
    type_ = txt.split('\n')[0]
    num_ = txt.split('\n')[-1]
    labels.append('%s\n%s\n%s' % (type_.split('.')[0], type_.split('.')[-1], num_))
rawdata_axes.set_xticklabels(labels)
fig.savefig('figures/IT_vs_PT_feat.png', bbox_inches='tight')
plt.close(fig)
