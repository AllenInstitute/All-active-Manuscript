#%% Visualizing Allen Insitute morphologies (.swc files)

from ateamopt.morph_handler import MorphHandler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ateam.data import lims
from ateamopt.utils import utility
from allensdk.core.cell_types_cache import CellTypesCache
import pandas as pd
import os

def get_morph_path(cell_id):
    lr = lims.LimsReader()
    morph_path = lr.get_swc_path_from_lims(int(cell_id))
    return morph_path 

data_path = os.path.join(os.getcwd(),os.pardir,os.pardir,'assets','aggregated_data')
cre_color_tasic16_filename = os.path.join(data_path,'cre_color_tasic16.pkl')
cre_color_dict = utility.load_pickle(cre_color_tasic16_filename)
cre_color_dict['Other'] = (0,0,0)

depth_data_filename = os.path.join(data_path,'mouse_me_and_met_avg_layer_depths.json') # Average layerwise depths for mouse
depth_data = utility.load_json(depth_data_filename)
total_depth = depth_data['wm']

# Cells are chosen to sample from diverse types within each layer
cell_id_dict = {
                #'1':['574734127','564349611','475585413','555341581','536951541'],
                '1':['574734127','475585413','536951541'],
#                '2/3':['485184849','475515168','485468180','476087653','571306690'],
                '2/3':['485184849','475515168','485468180'],
#                '4':['483101699','602822298','490205998','569723367','324257146'],
                '4':['483101699','602822298','569723367'],
                '5':['479225052','607124114','515249852'],
                '6a':['490259231','473564515','561985849'],
#                '6b':['589128331','574993444','510136749','509881736','590558808']
                '6b':['589128331']
                }

highlight_cells = ['485184849', '479225052', '473564515']

# Get normalized depth metadata for individual cells

ctc = CellTypesCache()
cells_allensdk = ctc.get_cells(species = ['Mus musculus'],simple = False)
sdk_data = pd.DataFrame(cells_allensdk)
sdk_data['specimen__id'] = sdk_data['specimen__id'].astype(str)

ylim_min,ylim_max =-200, 1200
soma_loc_x = 0
sigma_layer = 50
soma_loc_displacement_x = 350
unique_layers = sorted(sdk_data.structure__layer.unique().tolist())

layer_dist = {layer_ : i*soma_loc_displacement_x for i,layer_ in enumerate(unique_layers)}

sns.set(style='whitegrid')
fig,ax = plt.subplots()
figname = os.path.join('figures','morph_layerwise.png')
utility.create_filepath(figname)

for layer_name,cell_id_list in cell_id_dict.items():
    num_cells = len(cell_id_list)
    dist_x = np.linspace(-120,120,num_cells) # within layer x displacement
    for kk,cell_id in enumerate(cell_id_list):
        metadata= sdk_data.loc[sdk_data.specimen__id==cell_id,['csl__normalized_depth','line_name',
                                                               'structure__layer']]
        norm_depth = metadata.csl__normalized_depth.tolist()[0]
        cre = metadata.line_name.tolist()[0]
        layer = metadata.structure__layer.tolist()[0]
        color = cre_color_dict[cre] if cre in cre_color_dict.keys() else cre_color_dict['Other']
        color_dict = {swc_sect_indx:color for swc_sect_indx in range(1,5)}
    
        loc_x = layer_dist[layer]+dist_x[kk]
        soma_loc = np.array([loc_x,(1-norm_depth)*total_depth])
        soma_loc
        morph_path = get_morph_path(cell_id)
        
        morph_handler = MorphHandler(morph_path)
        morph_data,morph_apical,morph_axon,morph_dist_arr = morph_handler.get_morph_coords()
        
        # Rotate morphology to appear upright                            
        theta,axis_of_rot = morph_handler.calc_rotation_angle(morph_data,morph_apical)
#        if cell_id in highlight_cells:
#            lw,alpha,soma_rad = 1,1 ,20
#        else:
#            lw,alpha,soma_rad = .3,.8,20
        soma_rad = 20
        ax = morph_handler.draw_morphology_2D(theta,axis_of_rot,reject_axon=True,
                                  soma_loc=soma_loc,color_dict=color_dict,
                                  morph_dist_arr=morph_dist_arr,soma_rad = soma_rad,
                                  ax=ax)  # soma_rad = 10 in manuscript
        
        
ax.set_ylim([ylim_min,ylim_max])
xmin,xmax = ax.get_xlim()
for layer_ in unique_layers:
    layer_height = total_depth if layer_ == '1' else total_depth-depth_data[layer_] 
    ax.hlines(layer_height,xmin,xmax,colors='grey',lw=.5,linestyles='dashed')

# Add scale
#ax.plot([-50,50],[total_depth-depth_data['5'], total_depth-depth_data['5']],
#        lw=1,color='k')
#ax.text(0, total_depth-depth_data['5']-80, '$100\:\mu m$',fontsize=10,
#        horizontalalignment='center')
ax.plot([-50,-50],[total_depth-depth_data['5'], total_depth-depth_data['5']+200],
        lw=1,color='k')
ax.text(-110, total_depth-depth_data['5']+100, '$200\:\mu m$',fontsize=10,
        verticalalignment='center',rotation=90)

ax.hlines(0,xmin,xmax,colors='grey',lw=.25,linestyles='dashed')
ax.axis('off')
fig.set_size_inches(11,7) # (14,5) in the manuscript
fig.savefig(figname,bbox_inches='tight',dpi=500)

plt.close(fig)

#%% Plot ephys data

highlight_ephys_dict = {'485184849' :[32,45,59], '479225052' : [27,33, 41],
                        '473564515' : [24,34, 45]}
sns.set(style = 'whitegrid')
fig,ax = plt.subplots(len(highlight_ephys_dict),1,squeeze= False, figsize=(3,7), sharex = True)

for ii, cell_id in enumerate(highlight_ephys_dict.keys()):
    for sweep in highlight_ephys_dict[cell_id]:
        v, i, t = lims.get_sweep_v_i_t_lims(cell_id,sweep)
        metadata= sdk_data.loc[sdk_data.specimen__id==cell_id,'line_name']
        cre = metadata.tolist()[0]
        color = cre_color_dict[cre] if cre in cre_color_dict.keys() else cre_color_dict['Other']
        t *= 1e3
        i_non_zero = np.where(i != 0)
        i_start, i_end = i_non_zero[0][0], i_non_zero[0][-1]
        t_start, t_end = t[i_start], t[i_end]
        
        ax[ii,0].plot(t,v,color=color)
    ax[ii,0].set_xlim([t_start-50, t_end+50])
#    ax[ii,0].grid(False)
#    sns.despine(ax = ax[ii,0])
    ax[ii,0].axis('off')
    
fig.savefig('figures/select_ephys.png',bbox_inches='tight')
plt.close(fig)