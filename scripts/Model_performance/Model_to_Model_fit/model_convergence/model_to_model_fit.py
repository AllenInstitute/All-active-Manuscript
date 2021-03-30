import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import man_opt
import man_opt.utils as man_utils
from ateamopt.utils import utility
import pandas as pd
import numpy as np
from man_opt.utils import replace_channel_name


def shorten_channel_name(param_name):
    sec = param_name.split('.')[-1]
    shortened_name = replace_channel_name(param_name)
    if sec in shortened_name:
        return shortened_name
    else:
        return shortened_name + f'.{sec}'

# %% Model paths


data_path = Path(man_opt.__file__).parent.parent.joinpath(
    'assets/aggregated_data')
mouse_data_filename = Path(data_path / 'Mouse_class_data.csv')
mouse_datatype_filename = Path(data_path / 'Mouse_class_datatype.csv')
mouse_data_df = man_utils.read_csv_with_dtype(
    mouse_data_filename, mouse_datatype_filename)

cell_list = [
    "327962063",
    "473564515",
    "478793814",
    "481001895",
    "483101699",
    "484635029",
    "485058595",
    "486560376",
    "584682764"
]
cell_type_df = mouse_data_df.loc[(mouse_data_df.hof_index == 0) &
                                 (mouse_data_df.Cell_id.isin(cell_list)),
                                 ["Cell_id", "dendrite_type"]].reset_index(drop=True)

original_model_path = "/allen/aibs/mat/ateam_shared/Mouse_Model_Fit_Metrics"
model_refit_path = "/allen/programs/celltypes/workgroups/humancolumn_ephysmodeling/"\
    "anin/Optimizations_HPC/Mouse_Benchmark"


# %% Load Models
model_refit_list = []
original_model_list = []

for cell_id in cell_list:
    try:
        refit_path = os.path.join(model_refit_path, cell_id, "benchmark_final",
                                  f"Stage2/fitted_params/optim_param_{cell_id}_bpopt.json")
        model_refit_dict = utility.load_json(refit_path)
    except FileNotFoundError:
        refit_path = os.path.join(model_refit_path, cell_id, "benchmark_new",
                                  f"Stage2/fitted_params/optim_param_{cell_id}_bpopt.json")
        model_refit_dict = utility.load_json(refit_path)
    model_refit_dict.update({"Cell_id": cell_id})
    model_refit_list.append(model_refit_dict)

    original_path = os.path.join(original_model_path,
                                 f"{cell_id}/fitted_params/optim_param_{cell_id}_bpopt.json")
    original_model_dict = utility.load_json(original_path)
    original_model_dict.update({"Cell_id": cell_id})
    original_model_list.append(original_model_dict)

original_model_df = pd.DataFrame(original_model_list)
original_model_df["model_type"] = "Input Model"
refit_model_df = pd.DataFrame(model_refit_list)
refit_model_df["model_type"] = "Reoptimized Model"

all_model_df = pd.concat([original_model_df, refit_model_df])
all_model_df = pd.merge(all_model_df, cell_type_df, on="Cell_id")
params_melt = pd.melt(all_model_df, id_vars=["Cell_id", "dendrite_type", "model_type"],
                      var_name="param", value_name="value")
params_melt.value = np.abs(params_melt.value)
params_melt['param'] = params_melt['param'].apply(shorten_channel_name)
params_melt["section"] = params_melt["param"].apply(lambda x: x.split(".")[-1])

# %% Visualization

params_melt = params_melt.loc[~params_melt.section.isin(["basal"]), ]
params_group = params_melt.groupby("param")["value"].mean().\
    reset_index(name="value_avg").sort_values(by="value_avg")
section_list = list(utility.bpopt_section_map_inv.keys())
cond_params = params_melt["param"].unique().tolist()
cond_params_sorted = []
for section in section_list:
    params_section = [param_ for param_ in cond_params if param_.split(".")[-1] == section]
    cond_params_sorted.extend(sorted(params_section))

sns.set(style="whitegrid", font_scale=1.2)
g = sns.FacetGrid(params_melt, row="dendrite_type", despine=False)
g.map(sns.stripplot, "param", "value", "model_type", dodge=True,
      order=cond_params_sorted,
      size=5, marker="D", palette="Set1")
axes = g.axes.ravel()
for ax in axes:
    ax.set_yscale('log')
    ax.set_ylim((1e-10, 1e4))
    for i in range(len(cond_params_sorted)):
        ax.axvline(i, color="grey", linewidth=0.5)
g.set_axis_labels("", "Absolute parameter values")
plt.setp(axes[-1].get_xticklabels(), rotation=90)
handles, labels = axes[-1].get_legend_handles_labels()
axes[-1].legend(handles=handles, labels=labels, frameon=False, loc="lower right")

g.fig.set_size_inches(22, 8)
figname = 'figures/param_values_refitted.svg'
g.fig.savefig(figname, bbox_inches="tight")
