import pickle
import os
from ateamopt.bpopt_evaluator import Bpopt_Evaluator
import bluepyopt as bpopt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ateamopt.utils import utility
import json
from collections import defaultdict
from man_opt.utils import replace_channel_name


def shorten_channel_name(param_name):
    sec = param_name.split('.')[-1]
    shortened_name = replace_channel_name(param_name)
    if sec in shortened_name:
        return shortened_name
    else:
        return shortened_name + f'.{sec}'


def error_heatmap(obj_filename, figname, ignore_feat=[]):
    hof_obj_all_list = pickle.load(open(obj_filename, 'rb'))
    feature_list = list(
        set(list(map(lambda x: x.split('.')[-1], hof_obj_all_list[0].keys()))))
    ignore_feat.append('index')
    hof_df_list = []
    for i, hof_score in enumerate(hof_obj_all_list):
        obj_dict = {'index': i}
        for feature in feature_list:
            temp_list = list(map(lambda x: hof_score[x] if x.split('.')[-1] == feature else None,
                                 hof_score.keys()))

            temp_list_filtered = list(
                filter(lambda x: x is not None, temp_list))
            obj_dict[feature] = np.mean(temp_list_filtered)
        hof_df_list.append(obj_dict)

    hof_df = pd.DataFrame(hof_df_list)
    hof_df_obj = hof_df.loc[:, ~hof_df.columns.isin(ignore_feat)]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    sns.heatmap(hof_df_obj.T, cmap='viridis', ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    ax.set_xlabel('Hall of Fame')
    utility.create_filepath(figname)
    fig.savefig(figname, bbox_inches='tight')
    plt.close(fig)


hof_model_params_filename = os.path.join('data', 'hof_model_params.pkl')
hof_obj_train_filename = os.path.join('data', 'hof_obj_train.pkl')
original_model_filename = os.path.join('fitted_params_original',
                                       'optim_param_468193142_bpopt.json')

hof_model_params = pickle.load(open(hof_model_params_filename, 'rb'))
hof_obj_train = pickle.load(open(hof_obj_train_filename, 'rb'))
original_model_params = json.load(open(original_model_filename, 'r'))
original_model_params.update({'hof_index': 0})

protocols_path = os.path.join('config', 'train_protocols.json')
features_path = os.path.join('config', 'train_features.json')
morph_path = 'reconstruction.swc'
param_path = os.path.join('config', 'parameters.json')
mech_path = os.path.join('config', 'mechanism.json')

eval_handler = Bpopt_Evaluator(protocols_path, features_path,
                               morph_path, param_path, mech_path)

evaluator = eval_handler.create_evaluator()
opt = bpopt.optimisations.DEAPOptimisation(evaluator=evaluator)

param_names = opt.evaluator.param_names
hof_model_params_df = pd.DataFrame(hof_model_params, columns=param_names)
hof_model_params_df['hof_index'] = np.arange(40)
hof_model_params_df['status'] = 'prev'
param_original_df = pd.DataFrame(original_model_params, index=[0])
param_original_df['status'] = 'new'
param_df = pd.concat([hof_model_params_df, param_original_df], axis=0)

params_melt = pd.melt(param_df, id_vars=['hof_index', 'status'],
                      var_name='param', value_name='value')
params_melt.value = np.abs(params_melt.value)
params_melt['param'] = params_melt['param'].apply(shorten_channel_name)

sns.set(style='whitegrid', font_scale=1.4)

figname = 'figures/param_values.svg'
utility.create_filepath(figname)
fig, ax = plt.subplots(figsize=(10, 4))
markers = {"prev": "o", "new": "X"}

ax = sns.scatterplot(x='param', y='value', data=params_melt.loc[(params_melt.hof_index != 0) &
                                                                (params_melt.status == 'prev'), ], s=20,
                     linewidth=0, edgecolor='k', ax=ax, alpha=.5,
                     hue='param')
ax = sns.scatterplot(x='param', y='value', data=params_melt.loc[(params_melt.hof_index == 0) &
                                                                (params_melt.status == 'prev'), ], s=100,
                     linewidth=1, edgecolor='k', ax=ax, alpha=1,
                     hue='param', markers=markers)
ax = sns.scatterplot(x='param', y='value', data=params_melt.loc[(params_melt.hof_index == 0) &
                                                                (params_melt.status == 'new'), ], s=75,
                     ax=ax, alpha=1, color='k', marker="x", linewidth=1.5)
ax.set_yscale('log')
ax.set_ylim((1e-10, 1e4))
sns.despine(ax=ax)
ax.grid(False)
# Fake scatter plots
sc1 = ax.scatter([0], [1e5], marker='x', s=75, color='k', label='Input Model')
sc2 = ax.scatter([0], [1e5], marker='o', s=100,
                 color='k', label='Reoptimized Model')
ax.legend(handles=[sc1, sc2], frameon=False)
plt.setp(ax.get_xticklabels(), rotation=90)
ax.set_ylabel('Absolute Parameter Values')
ax.set_xlabel('')
fig.savefig(figname, bbox_inches='tight')
plt.close(fig)

hof_best_obj = defaultdict(list)
for key, val in hof_obj_train[0].items():
    hof_best_obj[key.split('.')[-1]].append(val)

# %% Error heatmap

heatmap_figname = 'figures/train_error_heatmap.pdf'
error_heatmap(hof_obj_train_filename, heatmap_figname,
              ignore_feat=['voltage_base'])


# %% Response comparison

stim_names = ['LongDC_26', 'LongDC_38', 'LongDC_50', 'LongDC_45']

model_resp_path = os.path.join('data', 'resp_opt.txt')
model_resp = pickle.load(open(model_resp_path, 'rb'))[0]
figname = os.path.join('figures', 'response_comparison.eps')
fig, ax = plt.subplots(1, 3, figsize=(10, 4))
for ii, stim_name in enumerate(stim_names):

    if ii <= 1:
        ax_ = ax[0]
        ax_.set_ylabel('Voltage (mV)')
    else:
        ax_ = ax[ii - 1]
        if ii == 2:
            ax_.set_xlabel('Time (ms)')
    exp_datapath = os.path.join('data', f'{stim_name}.txt')
    exp_data = np.loadtxt(exp_datapath)
    loc_name = f'{stim_name}.soma.v'

    ax_.plot(exp_data[:, 0], exp_data[:, 1], 'k', label='Input Model')
    ax_.plot(model_resp[loc_name]['time'], model_resp[loc_name]['voltage'],
             'b', label='Reoptimized Model')
    ax_.set_xlim([200, 1360])
    ax_.grid(False)
    sns.despine(ax=ax_)

h, l = ax_.get_legend_handles_labels()
fig.subplots_adjust(wspace=.3)
fig.legend(h[-2:], l[-2:], ncol=2, loc='lower center', frameon=False)
fig.tight_layout(rect=[0, 0.1, 1, .9])
fig.savefig(figname, bbox_inches='tight')
plt.close(fig)

# %% Spike shape and fi curve comparison
fi_exp_data = pickle.load(
    open(os.path.join('data', 'fI_exp_468193142.pkl'), 'rb'))
fi_model_data = pickle.load(
    open(os.path.join('data', 'fI_aa_468193142.pkl'), 'rb'))

ss_exp_data = pickle.load(
    open(os.path.join('data', 'AP_shape_exp_468193142.pkl'), 'rb'))
ss_model_data = pickle.load(
    open(os.path.join('data', 'AP_shape_aa_468193142.pkl'), 'rb'))

figname = os.path.join('figures', 'fi_ss_comp.svg')
fig, ax = plt.subplots(1, 3, figsize=(10, 4), gridspec_kw={
                       'width_ratios': [1, .5, .5]})
ax[0].plot(np.array(fi_exp_data['stim_exp'])*1e3,
           fi_exp_data['freq_exp'], 'k', marker='o')
ax[0].plot(np.array(fi_model_data['stim_All-active'])*1e3,
           fi_model_data['freq_All-active'], 'b', marker='o')

stim_names = ['LongDC_40', 'LongDC_46']
for ii, stim_name in enumerate(stim_names):
    ax[ii + 1].plot(ss_exp_data['time'], ss_exp_data[stim_name], 'k')
    ax[ii + 1].plot(ss_model_data['time'], ss_model_data[stim_name], 'b')

for jj, ax_ in enumerate(ax):
    ax_.grid(False)
    sns.despine(ax=ax_)
    if jj == 0:
        ax_.set_xlabel('Stimulus (pA)')
        ax_.set_ylabel('Spikecount')
    elif jj == 1:
        ax_.set_ylabel('Voltage (mV)')
        ax_.set_xlabel('Time (ms)')
    else:
        ax_.set_xlabel('Time (ms)')

fig.subplots_adjust(wspace=.3)
fig.savefig(figname, bbox_inches='tight')
plt.close(fig)

# %%
