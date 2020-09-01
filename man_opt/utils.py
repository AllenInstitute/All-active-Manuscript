import pandas as pd
from ateamopt.utils import utility
import numpy as np
import re
import requests


def read_csv_with_dtype(data_filename, datatype_filename):
    datatypes = pd.read_csv(datatype_filename)['types']
    data = pd.read_csv(data_filename, dtype=datatypes.to_dict())
    return data


def save_csv_with_dtype(data, data_filename, datatype_filename):
    data.dtypes.to_frame('types').to_csv(datatype_filename)
    data.to_csv(data_filename, index=None)


def get_data_fields(data_path):
    if isinstance(data_path, pd.DataFrame):
        return list(data_path)
    else:
        if data_path.endswith('.json'):
            json_data = utility.load_json(data_path)
            return list(json_data.keys())
        elif data_path.endswith('.csv'):
            csv_data = pd.read_csv(data_path, index_col=None)
            return list(csv_data)

    print('Not in .json,.csv or pandas dataframe format')
    return None


def get_cretype_cluster(cre_line, cre_cluster_dict):
    if cre_line in cre_cluster_dict.keys():
        return cre_cluster_dict[cre_line]
    else:
        return cre_line


def prepare_data_clf(data, feature_fields, target_field,
                     property_fields=[],
                     least_pop=5):

    data = data.loc[:, ~data.columns.duplicated()]
    data_section = data.loc[:, feature_fields +
                            property_fields + [target_field]]

    # drop any cell with target field nan
    data_section = data_section.dropna(axis=0, how='any',
                                       subset=[target_field] + property_fields)

    # filtering based on least populated class
    agg_data = data_section.groupby(
        target_field)[feature_fields[0]].agg(np.size).to_dict()
    filtered_targets = [key for key,
                        val in agg_data.items() if val >= least_pop]
    data_section = data_section.loc[data_section[target_field].isin(
        filtered_targets), ]

    # drop any feature which is nan for any cells
    data_section = data_section.dropna(axis=1, how='any')
    revised_features = [feature_field for feature_field in list(
        data_section) if feature_field in feature_fields]
    X_df = data_section.loc[:, revised_features + property_fields]
    y_df = data_section.loc[:, [target_field]]
    return X_df, y_df, revised_features


def replace_channel_name(param_name_):
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
    elif bool(re.search('gamma', param_name_)):
        param_name_ = 'gammaCa'
    elif bool(re.search('decay', param_name_)):
        param_name_ = 'decayCa'
    elif bool(re.search('Ca_LVA', param_name_)):
        param_name_ = 'CaLV'
    elif bool(re.search('Ca_HVA', param_name_)):
        param_name_ = 'CaHV'
    elif bool(re.search('SK', param_name_)):
        param_name_ = 'SK'
    elif bool(re.search('Ih', param_name_)):
        param_name_ = 'Ih'
    elif bool(re.search('Im', param_name_)):
        param_name_ = 'Im'
    return param_name_


def bounding_box(X):
    xmin, xmax = min(X, key=lambda a: a[0])[0], max(X, key=lambda a: a[0])[0]
    ymin, ymax = min(X, key=lambda a: a[1])[1], max(X, key=lambda a: a[1])[1]
    return (xmin, xmax), (ymin, ymax)


def pval_to_sig(pval):
    if pval < 1e-3:
        sig = '***'
    elif pval < 1e-2:
        sig = '**'
    elif pval < 5e-2:
        sig = '*'
    else:
        sig = 'n.s.'
    return sig


def annotate_sig_level(var_levels, hue_levels, hue_var, sig_group, sig_group_var,
                       plot_data, plot_data_x, plot_data_y, ax, **kwargs):
    y_sig_dict = {}
    for ii, var_level in enumerate(var_levels):
        whisk_max = -np.inf
        for hue_ in hue_levels:
            data = plot_data.loc[(plot_data[plot_data_x] == var_level) & (
                plot_data[hue_var] == hue_), plot_data_y]
            data = data.dropna(how='any', axis=0)
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            whisk_ = np.percentile(data, 75) + .75 * iqr
            whisk_max = whisk_ if whisk_ > whisk_max else whisk_max

        y_sig_dict[var_level] = whisk_max
    y_sig_max = np.max(list(y_sig_dict.values()))
    line_height_factor = kwargs.get('line_height_factor') or .05
    line_offset_factor = kwargs.get('line_offset_factor') or .2
    line_height = line_height_factor * y_sig_max
    line_offset = line_offset_factor * y_sig_max

    comp_sign = kwargs.get('comp_sign') or '<'
    for y_sig_key, y_sig_val in y_sig_dict.items():
        if y_sig_val != y_sig_max:
            y_sig_dict[y_sig_key] = y_sig_val + line_offset

    plot_type = kwargs.get('plot_type') or 'dabest'
    if plot_type == 'dabest':
        mid_hue_index = len(hue_levels)/2.0 - .5
        bar_width = 1
    else:
        mid_hue_index = len(hue_levels)/2.0
        bar_gap = 0.1
        bar_width = (1-2*bar_gap)/len(hue_levels)
    for param_, group in sig_group:
        if param_ not in var_levels:
            continue
        y_sig = y_sig_dict[param_]
        x_center = len(hue_levels)*var_levels.index(param_) + mid_hue_index

        for _, row_group in group.iterrows():
            cre_x, cre_y = row_group[sig_group_var].split(comp_sign)
            cre_x_idx = hue_levels.index(cre_x)
            cre_y_idx = hue_levels.index(cre_y)
            if plot_type == 'dabest':
                x_drift_x = abs(cre_x_idx - mid_hue_index)
                x_drift_y = abs(cre_y_idx - mid_hue_index)
            else:
                x_center = var_levels.index(param_)
                x_drift_x = abs(cre_x_idx - mid_hue_index + 0.5) * bar_width
                x_drift_y = abs(cre_y_idx - mid_hue_index + 0.5) * bar_width
            if cre_x_idx < cre_y_idx:
                x1, x2 = x_center - x_drift_x, x_center + x_drift_y
            else:
                x1, x2 = x_center + x_drift_x, x_center - x_drift_y

            ax.plot([x1, x1, x2, x2], [y_sig, y_sig + line_height, y_sig + line_height, y_sig],
                    lw=1, c='k')

            ax.text((x1 + x2) * .5, y_sig + .5 * line_height, row_group['sig_level'], ha='center',
                    va='bottom', color='k')

            y_sig += line_height
    return ax


# Download new all-active model for a specific cell-id


def getModel(cell_id, **kwargs):
    """
    Download an all-active model by cell_id using Allen Institute api
    """
    api_url = "http://api.brain-map.org"
    query_url = api_url + "/api/v2/data/query.json"

    # payload_all = {"criteria": "model::Specimen,rma::criteria,well_known_files(well_known_file_type[name$eq%s])" % """'UpdatedBiophysicalModelParameters'""",
    #                "num_rows": 1000}

    payload_specific = {"criteria": "model::Specimen,rma::criteria,[id$eq%d]" % cell_id,
                        "include": "well_known_files(well_known_file_type[name$eq%s])" % """'UpdatedBiophysicalModelParameters'"""}

    model_requests = requests.get(url=query_url, params=payload_specific)
    model_requests_json = model_requests.json()

    model_url = model_requests_json['msg'][0]['well_known_files'][0]['download_link']
    model_url = api_url + model_url
    model_params = requests.get(model_url)
    model_filename = f'{cell_id}/{cell_id}_fit.json' if not kwargs.get(
        'download_path') else kwargs['download_path']

    # Check if a new directory has to be created
    if len(model_filename.split('/')) > 1:
        utility.create_filepath(model_filename)
    with open(model_filename, 'wb') as json_file:
        json_file.write(model_params.content)
    return model_filename
