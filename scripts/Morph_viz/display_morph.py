from ateamopt.morph_handler import MorphHandler
import matplotlib.pyplot as plt
from ateam.data import lims
import seaborn as sns


def get_morph_path(cell_id):
    lr = lims.LimsReader()
    morph_path = lr.get_swc_path_from_lims(int(cell_id))
    return morph_path 

cell_id = '483101699'
morph_path = get_morph_path(cell_id)
morph_handler = MorphHandler(morph_path)
morph_data,morph_apical,morph_axon,morph_dist_arr = morph_handler.get_morph_coords()
theta,axis_of_rot = morph_handler.calc_rotation_angle(morph_data,morph_apical)

sns.set(style='whitegrid')
fig,ax = plt.subplots(figsize=(4,8))
ax = morph_handler.draw_morphology_2D(theta,axis_of_rot,reject_axon=False,ax=ax,
    axis_off=True,lw=3,alpha=1,soma_rad = 250,morph_dist_arr = morph_dist_arr)
fig.savefig('%s_morph.png'%cell_id,bbox_inches='tight',dpi=300)
plt.close(fig)
