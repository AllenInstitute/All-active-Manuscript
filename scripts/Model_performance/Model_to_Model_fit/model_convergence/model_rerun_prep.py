import requests
from allensdk.api.queries.biophysical_api import BiophysicalApi
from allensdk.core.cell_types_cache import CellTypesCache
import json
from collections import defaultdict
import sys
from ateamopt.utils import utility
import os
import logging

logging.basicConfig(level=logging.DEBUG)

# %% Download new all-active model for a specific cell-id


def getModel(cell_id, **kwargs):
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
    model_filename = f'{cell_id}/{cell_id}_fit.json'
    utility.create_filepath(model_filename)
    with open(model_filename, 'wb') as json_file:
        json_file.write(model_params.content)
    return model_filename


# %% Get ephys experiment file and the reconstruction


def getExperiment(cell_id, stimtypes):
    ctc = CellTypesCache()
    ephys_sweeps = ctc.get_ephys_sweeps(specimen_id=cell_id)
    sweeps_by_type = defaultdict(list)
    sweeps = []
    for sweep in ephys_sweeps:
        if sweep['stimulus_name'] in stimtypes:
            sweeps.append(sweep['sweep_number'])
            sweeps_by_type[sweep['stimulus_name']].append(
                sweep['sweep_number'])

    ephys_filename = f'{cell_id}/{cell_id}_ephys.nwb'
    ctc.get_ephys_data(cell_id, ephys_filename)
    swc_filename = f'{cell_id}/{cell_id}.swc'
    ctc.get_reconstruction(cell_id, swc_filename)
    return ephys_filename, swc_filename, sweeps, sweeps_by_type


# %% Create the config file to run the simulation


def prepareModelRun(model_filename, ephys_filename, swc_filename, sweeps,
                    sweeps_by_type):
    bp = BiophysicalApi()
    bp.create_manifest(fit_path=os.path.basename(model_filename),
                       model_type='Biophysical - all active',
                       stimulus_filename=os.path.basename(ephys_filename),
                       swc_morphology_path=os.path.basename(swc_filename),
                       sweeps=sweeps)
    bp.manifest['runs'][0]['sweeps_by_type'] = sweeps_by_type
    manifest_file = os.path.join(
        os.path.dirname(model_filename), 'manifest.json')
    with open(manifest_file, 'w') as manifest_json:
        json.dump(bp.manifest, manifest_json, indent=2)
    return manifest_file


if __name__ == '__main__':

    config_file = sys.argv[-1]
    try:
        from json.decoder import JSONDecodeError
        config = json.load(open(config_file, 'r'))
    except JSONDecodeError:
        logging.error('Need to pass a config file')
        raise
    model_filename = getModel(**config)
    ephys_filename, swc_filename, sweeps, sweeps_by_type = getExperiment(
        **config)
    prepareModelRun(model_filename, ephys_filename, swc_filename, sweeps,
                    sweeps_by_type)
