# Run the python file like this:
# On a different terminal : ipcluster start (in the same directory of the script)
# python ephys_data_collection.py -l "L5 CF" "L5 IT" -c ttype
# If need to debug run iopubwatcher.py from another terminal
# Don't need to run on HPC (for small enough cell ids)

import pandas as pd
import os
import man_opt.utils as man_utils
from ateamopt.utils import utility
from ipyparallel import Client
import logging
import man_opt
import argparse


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

parser = argparse.ArgumentParser(
    description='Get cell-type specific ephys features')
parser.add_argument('-l', '--cty_list', nargs='+',
                    help='List of types to be selected', required=True)
parser.add_argument('-c', '--cty', type=str, choices=['ttype', 'Cre_line'],
                    help='cell-type attribute to filter on', default='ttype')


def get_efeatures(cell_id):
    from ateamopt.nwb_extractor import NwbExtractor
    from ateamopt.optim_config_rules import correct_voltage_feat_std
    from ateam.data import lims
    import os
    from ateamopt.utils import utility
    import shutil

    acceptable_stimtypes = ['Long Square']
    feature_names_path = 'feature_set_all.json'
    ephys_dir_origin = 'ephys_ttype'
    efel_feature_path = 'eFEL_features_ttype'
    lr = lims.LimsReader()

    cell_efeatures_dir = os.path.join(efel_feature_path, cell_id)
    cell_protocols_filename = os.path.join(
        cell_efeatures_dir, 'protocols.json')
    cell_features_filename = os.path.join(
        efel_feature_path, cell_id, 'features.json')
    efeature_filenames = [cell_protocols_filename, cell_features_filename]

    if not all(os.path.exists(filename_) for filename_ in efeature_filenames):
        utility.create_filepath(cell_protocols_filename)
        nwb_path = lr.get_nwb_path_from_lims(
            int(cell_id), get_sdk_version=True)
        cell_ephys_dir = os.path.join(ephys_dir_origin, cell_id)
        nwb_handler = NwbExtractor(cell_id, nwb_path)
        ephys_data_path, stimmap_filename = nwb_handler.save_cell_data_web(
            acceptable_stimtypes, ephys_dir=cell_ephys_dir)

        protocol_dict, feature_dict = nwb_handler.get_efeatures_all(feature_names_path,
                                                                    ephys_data_path, stimmap_filename)
        feature_dict = correct_voltage_feat_std(feature_dict)

        utility.save_json(cell_protocols_filename, protocol_dict)
        utility.save_json(cell_features_filename, feature_dict)
        shutil.move(stimmap_filename, cell_efeatures_dir)
        shutil.rmtree(cell_ephys_dir, ignore_errors=True)


def Main():
    args = parser.parse_args()
    cty_type = args.cty
    cty_list = args.cty_list

    data_path = os.path.join(os.path.dirname(man_opt.__file__),
                             os.pardir, 'assets', 'aggregated_data')
    mouse_data_filename = os.path.join(data_path, 'Mouse_class_data.csv')
    mouse_datatype_filename = os.path.join(
        data_path, 'Mouse_class_datatype.csv')
    me_ttype_map_path = os.path.join(data_path, 'me_ttype.pkl')

    sdk_data_filename = os.path.join(data_path, 'sdk.csv')
    sdk_datatype_filename = os.path.join(data_path, 'sdk_datatype.csv')
    sdk_data = man_utils.read_csv_with_dtype(
        sdk_data_filename, sdk_datatype_filename)

    if cty_type == 'ttype':
        mouse_data = man_utils.read_csv_with_dtype(
            mouse_data_filename, mouse_datatype_filename)
        me_ttype_map = utility.load_pickle(me_ttype_map_path)

        metype_cluster = mouse_data.loc[mouse_data.hof_index == 0, [
            'Cell_id', 'Dendrite_type', 'me_type']]
        sdk_me = pd.merge(
            sdk_data, metype_cluster, how='left', on='Cell_id').dropna(how='any', subset=['me_type'])

        sdk_me['ttype'] = sdk_me['me_type'].apply(
            lambda x: me_ttype_map[x])
        cell_df = sdk_me.loc[sdk_me.ttype.isin(cty_list), ]
    elif cty_type == 'Cre_line':
        cell_df = sdk_data.loc[sdk_data.line_name.isin(cty_list), ]

    cell_ids = cell_df.Cell_id.unique().tolist()
    rc = Client(profile=os.getenv('IPYTHON_PROFILE'))
    logger.debug('Using ipyparallel with %d engines', len(rc))
    lview = rc.load_balanced_view()
    lview.map_sync(get_efeatures, cell_ids)


if __name__ == '__main__':
    Main()
