from ateamopt.morph_handler import MorphHandler, swc_dict
import matplotlib.pyplot as plt
from ateam.data import lims
import seaborn as sns


def get_morph_path(cell_id):
    lr = lims.LimsReader()
    morph_path = lr.get_swc_path_from_lims(int(cell_id))
    return morph_path


cell_id = '479225052'
morph_path = get_morph_path(cell_id)
morph_handler = MorphHandler(morph_path)
morph_data, morph_apical, morph_axon, morph_dist_arr = morph_handler.get_morph_coords()
theta, axis_of_rot = morph_handler.calc_rotation_angle(morph_data, morph_apical)
swc_sect_indices = list(swc_dict.keys())
color_dict = {swc_sect_indx: 'crimson' for swc_sect_indx in swc_sect_indices}

sns.set(style='whitegrid')
ax, elev_angle = morph_handler.draw_morphology(theta, axis_of_rot, reject_axon=False,
                                               morph_dist_arr=morph_dist_arr, axis_off=True,
                                               color_dict=color_dict, lw=0.5)
n_syn_apical = 15
ax = morph_handler.add_synapses(morph_apical, n_syn_apical, theta, axis_of_rot, ax, color='k')
ax.figure.savefig('figures/%s_morph.png' % cell_id, bbox_inches='tight', dpi=300)
plt.close(ax.figure)