import pandas as pd
from ateamopt.utils import utility
import numpy as np
import re
from sklearn import metrics
import numpy as np

def read_csv_with_dtype(data_filename,datatype_filename):
    datatypes = pd.read_csv(datatype_filename)['types']
    data = pd.read_csv(data_filename, dtype=datatypes.to_dict())
    return data

def save_csv_with_dtype(data,data_filename,datatype_filename):
    data.dtypes.to_frame('types').to_csv(datatype_filename)
    data.to_csv(data_filename, index=None)


def get_data_fields(data_path):
    if isinstance(data_path,pd.DataFrame):
        return list(data_path)
    else:
        if data_path.endswith('.json'):
            json_data = utility.load_json(data_path)
            return list(json_data.keys())
        elif data_path.endswith('.csv'):
            csv_data = pd.read_csv(data_path, index_col = None)
            return list(csv_data)

    print('Not in .json,.csv or pandas dataframe format')
    return None


def prepare_data_clf(data,feature_fields,target_field,
                         property_fields = [],
                         least_pop=5):

    data = data.loc[:,~data.columns.duplicated()]
    data_section = data.loc[:,feature_fields+property_fields+\
                            [target_field]]

    # drop any cell with target field nan
    data_section = data_section.dropna(axis=0, how = 'any',
                           subset=[target_field] + property_fields)

    # filtering based on least populated class
    agg_data = data_section.groupby(target_field)[feature_fields[0]].\
                    agg(np.size).to_dict()
    filtered_targets = [key for key,val in agg_data.items() \
                        if val >= least_pop]
    data_section = data_section.loc[data_section[target_field].\
                    isin(filtered_targets),]

    # drop any feature which is nan for any cells
    data_section = data_section.dropna(axis =1,
                                       how = 'any')
    revised_features = [feature_field for feature_field in \
                list(data_section) if feature_field in feature_fields]
    X_df = data_section.loc[:,revised_features + property_fields]
    y_df = data_section.loc[:,[target_field]]
    return X_df,y_df,revised_features


def replace_channel_name(param_name_):
    if bool(re.search('NaT',param_name_)):
            param_name_ = 'NaT'
    elif bool(re.search('Nap',param_name_)):
        param_name_ = 'NaP'
    elif bool(re.search('K_P',param_name_)):
        param_name_ = 'KP'
    elif bool(re.search('K_T',param_name_)):
        param_name_ = 'KT'
    elif bool(re.search('Kv3_1',param_name_)):
        param_name_ = 'Kv31'
    elif bool(re.search('gamma',param_name_)):
        param_name_ = 'gammaCa'
    elif bool(re.search('decay',param_name_)):
        param_name_ = 'decayCa'
    elif bool(re.search('Ca_LVA',param_name_)):
        param_name_ = 'CaLV'
    elif bool(re.search('Ca_HVA',param_name_)):
        param_name_= 'CaHV'
    elif bool(re.search('SK',param_name_)):
        param_name_= 'SK'
    elif bool(re.search('Ih',param_name_)):
        param_name_= 'Ih'

    return param_name_


def silhouette_score(estimator, X):
    cluster_labels = estimator.fit_predict(X)
    num_labels = len(set(cluster_labels))
    num_samples = X.shape[0]
    if num_labels == 1 or num_labels == num_samples:
        return -1
    else:
        return metrics.silhouette_score(X, cluster_labels)


def bounding_box(X):
    xmin, xmax = min(X,key=lambda a:a[0])[0], max(X,key=lambda a:a[0])[0]
    ymin, ymax = min(X,key=lambda a:a[1])[1], max(X,key=lambda a:a[1])[1]
    return (xmin,xmax), (ymin,ymax)


def gap_statistic(estimator, X):
    np.random.seed(0)
    (xmin,xmax), (ymin,ymax) = bounding_box(X)
    n_references=1
    ref_inertia_rep = []
    for _ in range(n_references):
        ref_data = np.random.uniform(low=(xmin,ymin),high=(xmax,ymax),
                                     size=X.shape)
        estimator.fit(ref_data)
        ref_inertia_rep.append(estimator.inertia_)
    ref_inertia = np.mean(np.log(ref_inertia_rep))
    ref_std = np.std(np.log(ref_inertia_rep))
    estimator.fit(X)
    local_inertia = estimator.inertia_
    gap = ref_inertia-np.log(local_inertia)
    sk = ref_std*np.sqrt(1+1/n_references)
    return gap,sk    