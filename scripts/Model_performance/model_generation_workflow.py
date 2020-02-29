import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from ateamopt.utils import utility
import seaborn as sns
import matplotlib.ticker as ticker

cell_id = '483101699'
model_path = '/allen/aibs/mat/anin/hpc_trials/BluePyOpt_speedup/483101699'

stage_trace = {'Stage0': ['LongDC_39','LongDC_43'],
               'Stage1': ['LongDC_39','LongDC_43'],
               'Stage2': ['LongDC_39','LongDC_43','LongDC_51']}


model_color = (128/255.,128/255., 128/255.)

sns.set(style='whitegrid')
fig,ax = plt.subplots(1,len(stage_trace.keys()),figsize=(8,2))

for ii,stage in enumerate(stage_trace.keys()):
    stage_path = os.path.join(model_path,stage)
    model_resp_path = os.path.join(stage_path,'resp_opt.txt')
    model_resp = utility.load_pickle(model_resp_path)[0]
    
    for stim in stage_trace[stage]:
        exp_path = os.path.join(stage_path,'preprocessed/%s.txt'%stim)
        exp = np.loadtxt(exp_path)
        ax[ii].plot(exp[:,0],exp[:,1],'k',label='Experiment')
        
        model_resp_stim = model_resp['%s.soma.v'%stim]
        ax[ii].plot(model_resp_stim['time'],model_resp_stim['voltage'],color=model_color,
          label='Model')
        

for ax_ in ax:
    ax_.grid(False)
    ax_.set_xlim([200,1400])
    
    if ax_ != ax[-1]:
        ax_.set_ylim([-101,-85])
        sns.despine(ax=ax_,bottom=True)
        ax_.spines['left'].set_bounds(-90, -85)
        ax_.yaxis.set_major_locator(ticker.FixedLocator([-90,-85]))
        ax_.set_xticklabels('')
    else:
        sns.despine(ax=ax_)
        ax_.spines['bottom'].set_bounds(500, 1000)
        ax_.xaxis.set_major_locator(ticker.FixedLocator([500, 1000]))
        ax_.spines['left'].set_bounds(-50, 0)
        ax_.yaxis.set_major_locator(ticker.FixedLocator([-50,0]))
        h,l = ax_.get_legend_handles_labels()
        h,l = h[-2:],l[-2:]
fig.subplots_adjust(hspace=0)
fig.legend(h,l,loc='lower center',ncol=2,frameon=False)
fig.tight_layout(rect=[0,.05,1,.95])        
fig.savefig('resp_comparison.pdf',bbox_inches='tight')
plt.close(fig)


#%% Evolution of the objectives

stages = ['Stage0', 'Stage1', 'Stage2']

sns.set(style='whitegrid')
fig,ax= plt.subplots(1,len(stages),figsize=(8,2))

for jj,stage in enumerate(stages):
    ga_evol_params_path = os.path.join(model_path,stage,'analysis_params',
                                       'GA_evolution_params.pkl')
    ga_evol_params = utility.load_pickle(ga_evol_params_path)
    gen_numbers = ga_evol_params['gen_numbers']
    mean = ga_evol_params['mean']
    std = ga_evol_params['std']
    minimum = ga_evol_params['minimum']
    
    stdminus = mean - std
    stdplus = mean + std
    
    
    ax[jj].plot(gen_numbers,
                    mean,color='firebrick',
                    linewidth=2,
                    label='Population Avg')
    
    ax[jj].fill_between(gen_numbers,
        stdminus,
        stdplus,
        color='salmon',
        alpha = 0.5)
    
    ax[jj].plot(gen_numbers,
            minimum,
            color=model_color,
            linewidth=2,
            alpha = 0.8,
            label='Population Minimum')
    ax[jj].set_ylim([0,np.max(stdplus+100)])
    if jj == 2:
        ax[jj].set_xlim([0,200])
    else:
        ax[jj].set_xlim([0,50])
    ax[jj].grid(False)
    sns.despine(ax=ax[jj])
    ax[jj].set_xlabel('# Generations')
    
ax[1].legend(frameon=False)      
ax[0].set_ylabel('Sum of objectives') 
fig.subplots_adjust(wspace=.3)           
fig.savefig('ga_evolution.pdf',bbox_inches='tight')
plt.close(fig)

#%% Model Validation

fig,ax = plt.subplots(2,1,figsize=(4,5),sharex=True)
noise_trace = 'Noise_2_62'
exp_path_noise = os.path.join(stage_path,'preprocessed/%s.txt'%noise_trace)
exp = np.loadtxt(exp_path_noise)
ax[0].plot(exp[:,0], exp[:,2],'maroon')
ax[1].plot(exp[:,0],exp[:,1],'k')
model_resp_stim = model_resp['%s.soma.v'%noise_trace]
ax[1].plot(model_resp_stim['time'],model_resp_stim['voltage'],color=model_color)

ax[1].spines['bottom'].set_bounds(1000, 6000)
ax[1].xaxis.set_major_locator(ticker.FixedLocator([1000, 6000]))
ax[1].spines['left'].set_bounds(-50, 0)
ax[1].yaxis.set_major_locator(ticker.FixedLocator([-50,0]))

for ax_ in ax:
    ax_.grid(False)
    ax_.set_xlim([200,22000])
    sns.despine(ax = ax_)
fig.savefig('Noise_resp.pdf',bbox_inches='tight')
plt.close(fig)



#%% Feature descriptions

#fig,ax = plt.subplots(1,4,figsize=(14,3),gridspec_kw={'width_ratios':[1,1,.7,2]})
#exp_path_depol_sub = os.path.join(stage_path,'preprocessed/LongDC_43.txt')
#exp = np.loadtxt(exp_path_depol_sub)
#ax[0].plot(exp[:,0],exp[:,1],'k')
#ax[0].set_xlim([0,1600])
#
#exp_path_hyperpol_sub = os.path.join(stage_path,'preprocessed/LongDC_39.txt')
#exp = np.loadtxt(exp_path_hyperpol_sub)
#ax[1].plot(exp[:,0],exp[:,1],'k')
#ax[1].set_xlim([0,1800])
#
#spike_path = os.path.join(model_path,'Stage2','Validation_Responses',
#                                   'AP_shape_exp_483101699.pkl')
#spike_resp = utility.load_pickle(spike_path)
#ax[2].plot(spike_resp['time'],spike_resp['LongDC_55'],'k',lw=2)
#
#exp_path_spike = os.path.join(stage_path,'preprocessed/LongDC_51.txt')
#exp = np.loadtxt(exp_path_spike)
#ax[3].plot(exp[:,0],exp[:,1],'k')
#ax[3].set_xlim([200,1400])
#
#for ax_ in ax:
#    ax_.axis('off')
#
#fig.savefig('feature_description.pdf',bbox_inches='tight')
#plt.close(fig)
#
#
#
