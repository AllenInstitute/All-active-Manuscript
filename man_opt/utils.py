import pandas as pd
from ateamopt.utils import utility
import numpy as np

def read_csv_with_dtype(data_filename,datatype_filename):
    datatypes = pd.read_csv(datatype_filename)['types']
    data = pd.read_csv(data_filename, dtype=datatypes.to_dict())
    return data

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
