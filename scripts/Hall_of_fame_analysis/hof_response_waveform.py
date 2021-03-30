import os
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from ateamopt.utils import utility
import matplotlib
matplotlib.use('Agg')


cell_id = '483101699'
model_data_path = ('/allen/aibs/mat/anin/hpc_trials/BluePyOpt_speedup/'
                   '%s/Stage2' % cell_id)
hof_responses_path = os.path.join(model_data_path, 'analysis_params/hof_response_all.pkl')
hof_responses = utility.load_pickle(hof_responses_path)

hof_num = 40
color_palette = sns.color_palette('Blues_r', hof_num)

select_stim_names = ['LongDC_55', 'Short_Square_Triple_101',
                     'Ramp_5', 'Noise_1_63']

sns.set_style('whitegrid')
fig, ax = plt.subplots(1, len(select_stim_names), sharey=True, squeeze=False,
                       gridspec_kw={'width_ratios': [1, 1, 1.5, 1.5]},
                       figsize=(12, 3.5))

hof_array_r = np.arange(hof_num - 1, -1, -1)
hof_responses_r = list(reversed(hof_responses))

for idx, select_stim in enumerate(select_stim_names):
    for ii, hof_response in zip(hof_array_r, hof_responses_r):
        time = hof_response[0]['%s.soma.v' % select_stim]['time']
        response = hof_response[0]['%s.soma.v' % select_stim]['voltage']
        ax[0, idx].plot(time, response, color=color_palette[ii], lw=1)
    if idx == 0:
        ax[0, idx].set_xlim([200, 1400])
        ax[0, idx].set_ylabel('Voltage $(mV)$', fontsize=12)
    elif idx == 1:
        ax[0, idx].set_xlim([1000, 2000])
    ax[0, idx].grid(False)
    ax[0, idx].set_xlabel('Time $(ms)$', fontsize=12)
    ax[0, idx].set_title(select_stim.rsplit('_', 1)[0], fontsize=12)
    sns.despine(ax=ax[0, idx])


my_cmap = ListedColormap(color_palette)
sm = ScalarMappable(cmap=my_cmap, norm=plt.Normalize(0, hof_num - 1))
sm.set_array([])

cbar = fig.colorbar(sm, fraction=0.05,
                    boundaries=np.arange(hof_num + 1) - 0.5)
cbar.set_ticks(np.arange(0, hof_num, 10))
cbar.ax.tick_params(axis=u'both', which=u'both', length=0)
cbar.ax.set_yticklabels(np.arange(0, hof_num, 10), fontsize=12)
cbar.set_label('Hall of Fame index', fontsize=12)
cbar.outline.set_visible(False)
fig.savefig('figures/hof_responses_waveform.svg', bbox_inches='tight')
plt.close(fig)
