import pandas as pd
import seaborn as sns
from ateamopt.utils import utility
import matplotlib.pyplot as plt
import os
import man_opt.utils as man_utils
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import mannwhitneyu
from matplotlib.ticker import FormatStrFormatter
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def draw_heatmap(figname,dist_matrix,patch_length=40):
    
    mask =  np.tri(dist_matrix.shape[0], k=0)
    dist_matrix= np.ma.array(dist_matrix, mask=mask)
    
    fontsize = 16
    fig_,ax_ = plt.subplots(figsize = (5,5))
    im_ = ax_.imshow(dist_matrix,vmin=0,cmap='viridis')
    
    ax_.set_xlabel('Hall of Fame Models',fontsize=fontsize)
    ax_.set_ylabel('Hall of Fame Models',fontsize=fontsize)
    ax_.set_title(bcre_,fontsize=fontsize)
    ax_.set_xticklabels(['']*len(ax_.get_xticklabels()))
    ax_.set_yticklabels(['']*len(ax_.get_yticklabels()))
    ax_.grid(False)
    current_idx = 0
    for _ in range(len(unique_cell_ids)):
        ax_.add_patch(PathPatch(Path([[current_idx, current_idx],
                                      [current_idx+patch_length, current_idx],
                                      [current_idx+patch_length, current_idx+patch_length],
                                      [current_idx,current_idx]]),
                                      facecolor="none",
                                      edgecolor='k',
                                      linewidth=2))
        
        current_idx += patch_length
    divider = make_axes_locatable(ax_)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar_ = plt.colorbar(im_,ax=ax_,cax=cax,ticks=[0,.25,.5])
    cax.tick_params(labelsize=fontsize)
    cbar_.outline.set_visible(False)
    cax.set_ylabel('Normalized distance',fontsize=fontsize)
    fig_.savefig(figname,bbox_inches='tight')
    
    plt.close(fig_)


def process_conductance_data(hof_param_data,filter_list = ['Cell_id','hof_index','Broad_Cre_line']):
    hof_param_data_bcre_ = hof_param_data.dropna(axis =1, how = 'any')
    conductance_params = [dens_ for dens_ in list(hof_param_data_bcre_) 
                          if dens_ not in filter_list]
    conductance_val_bcre_ = hof_param_data_bcre_.loc[:,conductance_params].values

    conductance_val_bcre_ = scaler.fit_transform(conductance_val_bcre_)/conductance_val_bcre_.shape[1]
    conductance_dist_matrix = euclidean_distances(conductance_val_bcre_)
    return hof_param_data_bcre_, conductance_dist_matrix

def draw_significance(mu_dist1,mu_dist2,pval,ax,height_offset = .1):
    sig_text = man_utils.pval_to_sig(pval)
    bin_dist1,_ = np.histogram(mu_dist1,density = True,bins=10)
    bin_dist2,_ = np.histogram(mu_dist2,density = True,bins=10)
    mean_dist1 = np.mean(mu_dist1)
    mean_dist2 = np.mean(mu_dist2)
    y_height1 = np.max(bin_dist1)
    y_height2 = np.max(bin_dist2)
    max_y_height = (1+height_offset)*np.max([y_height1, y_height2])
    ax.plot([mean_dist1, mean_dist1, mean_dist2, mean_dist2], [y_height1, max_y_height,max_y_height,y_height2],color= 'k')
    ax.text((mean_dist1+mean_dist2)*.5, max_y_height, sig_text, ha='center', 
                    va='bottom', color='k')
    return ax

#%% Data path
data_path = os.path.join(os.getcwd(),os.pardir,os.pardir,'assets','aggregated_data')
mouse_data_filename = os.path.join(data_path,'Mouse_class_data.csv')
mouse_datatype_filename = os.path.join(data_path,'Mouse_class_datatype.csv')
param_data_filename = os.path.join(data_path,'allactive_params.csv')
param_datatype_filename = os.path.join(data_path,'allactive_params_datatype.csv')
bcre_coloring_filename = os.path.join(data_path,'bcre_color_tasic16.pkl')

#%% Loading data

mouse_data_df = man_utils.read_csv_with_dtype(mouse_data_filename,mouse_datatype_filename)
hof_param_data = man_utils.read_csv_with_dtype(param_data_filename,param_datatype_filename)

bcre_cluster = mouse_data_df.loc[mouse_data_df.hof_index==0,['Cell_id','Broad_Cre_line']]
hof_param_data_bcre = pd.merge(hof_param_data,bcre_cluster,how='left',on='Cell_id')
bcre_color_dict = utility.load_pickle(bcre_coloring_filename)

bcre_order = list(bcre_color_dict.keys())

hof_param_data_bcre = hof_param_data_bcre.loc[hof_param_data_bcre.Broad_Cre_line.isin(bcre_order),]
hof_param_data_bcre.Broad_Cre_line = pd.Categorical(hof_param_data_bcre.Broad_Cre_line,
                                        categories=bcre_order) # Order according the bcre order
hof_param_data_bcre = hof_param_data_bcre.sort_values(['Broad_Cre_line','Cell_id','hof_index'])
hof_param_data_bcre.reset_index(drop=True,inplace=True)
hof_num = 40
intra_dist,inter_dist = {},{}
scaler = StandardScaler()


#%% Plotting the individual distance matrix for broad classes

sns.set_style('whitegrid')
color_pal = sns.color_palette()
intra_cell_intra_class_col, inter_cell_intra_class_col= color_pal[0],color_pal[1]

fig,ax = plt.subplots(2,len(bcre_order)//2,sharex=False,sharey=False,figsize=(5,5))

for ii,bcre_ in enumerate(bcre_order):
    hof_param_data_bcre_ = (hof_param_data_bcre.loc[hof_param_data_bcre.Broad_Cre_line == bcre_,])  
    hof_param_data_bcre_, conductance_dist_matrix = process_conductance_data(hof_param_data_bcre_)
    unique_cell_ids = hof_param_data_bcre_.Cell_id.unique()
    
    intra_mask_cell = np.triu(np.ones((hof_num,hof_num)),k=1)
    intra_mask_blkdiag = np.kron(np.eye(len(unique_cell_ids)),intra_mask_cell)
    
    inter_mask_cell = np.triu(np.ones((hof_num,hof_num)),k=1)
    inter_mask_blkdiag = np.kron(inter_mask_cell,np.ones((len(unique_cell_ids),
                                                          len(unique_cell_ids))))
    
    masked_intra_conductance = np.multiply(conductance_dist_matrix,intra_mask_blkdiag).flatten()
    masked_intra_conductance = masked_intra_conductance[masked_intra_conductance!=0]
    masked_inter_conductance = np.multiply(conductance_dist_matrix,inter_mask_blkdiag).flatten()
    masked_inter_conductance = masked_inter_conductance[masked_inter_conductance!=0]
    
    # plot the schema for calculation
#    if ii == 0:
#        schema_fig,(s_ax1,s_ax2) = plt.subplots(1,2)
#        s_ax1 = sns.heatmap(intra_mask_blkdiag,ax=s_ax1)
#        s_ax2 = sns.heatmap(inter_mask_blkdiag,ax=s_ax2)
#        
#        for s_ax in (s_ax1,s_ax2):
#            s_ax.axis('off')
#        schema_fig.savefig('figures/intra_inter_schema.pdf',bbox_inches='tight')
#        plt.close(schema_fig)
#    
    
    _,p_less_ = mannwhitneyu(masked_intra_conductance,masked_inter_conductance,alternative='less')
    
    sig_text = man_utils.pval_to_sig(p_less_)
    
    intra_dist[bcre_]=masked_intra_conductance.tolist()
    inter_dist[bcre_]=masked_inter_conductance.tolist()
    
    axis_fontsize = 14
    tick_fontsize = 12
    legend_fontsize =axis_fontsize
    ax[ii//2,ii%2] = sns.distplot(intra_dist[bcre_],norm_hist=True,ax=ax[ii//2,ii%2],hist_kws={'label':
            'intra'},color=intra_cell_intra_class_col)
    ax[ii//2,ii%2] = sns.distplot(inter_dist[bcre_],norm_hist = True,ax=ax[ii//2,ii%2],hist_kws={'label':
            'inter'},color=inter_cell_intra_class_col)
    ax[ii//2,ii%2].set_title(bcre_,fontsize=axis_fontsize)
    ax[ii//2,ii%2].text(.5,.9,sig_text,ha='center',va='center',
      fontsize= axis_fontsize,
      transform=ax[ii//2,ii%2].transAxes)
    ax[ii//2,ii%2].grid(False)
    sns.despine(ax=ax[ii//2,ii%2])
    ax[ii//2,ii%2].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    plt.setp(ax[ii//2,ii%2].get_xticklabels(),fontsize=tick_fontsize)
    plt.setp(ax[ii//2,ii%2].get_yticklabels(),fontsize=tick_fontsize)
    
    heatmap_figname = 'figures/intra_inter_distbn_heatmap_%s.svg'%bcre_
    draw_heatmap(heatmap_figname,conductance_dist_matrix)
    
h,l = ax[ii//2,ii%2].get_legend_handles_labels()
fig.legend(handles=h,labels=l,loc= 'lower center',ncol=2,frameon=False)
fig.subplots_adjust(wspace=.01)
fig.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.savefig('figures/intra_inter_distbn.svg',bbox_inches='tight')
plt.close(fig)


#%% Plot the inhibitory block (Htr3a, Sst, Pvalb)


plt_inh_dispersion = 0

if plt_inh_dispersion:
    inh_lines = ['Htr3a','Sst','Pvalb']
    hof_param_data_bcre_inh = hof_param_data_bcre.loc[hof_param_data_bcre.Broad_Cre_line.isin(inh_lines),]
    param_df_bcre,cond_inh = process_conductance_data(hof_param_data_bcre_inh)
    mask =  np.tri(param_df_bcre.shape[0], k=0)
    masked_cond_inh = np.ma.array(cond_inh, mask=mask)
    inh_palette = {inh_line_:bcre_color_dict[inh_line_] for inh_line_ in inh_lines}
    cre = param_df_bcre.loc[param_df_bcre.Broad_Cre_line.isin(inh_lines),'Broad_Cre_line'].astype(str)
    row_colors = cre.map(inh_palette)
    
    g = sns.clustermap(pd.DataFrame(masked_cond_inh),row_cluster=False,
                   col_cluster=False,row_colors=row_colors,cmap='viridis')
    
    ax = g.ax_heatmap
    ax.set_position([.1, .1, .8, .85])
    
    ax_row_color = g.ax_row_colors
    ax.grid(False)
    ax.axis('off')
    ax_row_color.set_position([.91, .1, .04, .85])
    ax_row_color.set_xticklabels('')
    g.cax.set_position([.1, .1, .02, .2])
    g.cax.tick_params(axis=u'both', which=u'both',length=0)
    
    
    current_idx = 0
    line_idx = 0
    
    all_cell_ids = param_df_bcre.Cell_id.unique().tolist()
    num_all_cells = len(all_cell_ids)
    all_hof_models = param_df_bcre.shape[0]
    
    avg_dist_mat = np.zeros((num_all_cells, num_all_cells))
    num_comparisons = len(inh_lines)*2
    
    for ii in range(num_all_cells):
        for jj in range(num_all_cells):
            chunk = cond_inh[ii*hof_num:(ii+1)*hof_num,jj*hof_num:(jj+1)*hof_num]
            avg_chunk = np.mean(chunk[np.triu_indices(hof_num,k=1)])
            avg_dist_mat[ii,jj] = avg_chunk
            
    for line in inh_lines:
        line_hof_models = param_df_bcre.loc[param_df_bcre.Broad_Cre_line == line,].shape[0]
        num_line_cells = int(line_hof_models/hof_num)
        
        intra_class_blkdiag = np.zeros(num_all_cells)
        intra_class_blkdiag[line_idx:line_idx+num_line_cells] =1
        intra_class_blkdiag = np.diag(intra_class_blkdiag)
        
        intra_class_intra_cell_blkdiag = np.zeros((num_all_cells,num_all_cells))
        intra_class_intra_cell_blkdiag[line_idx:line_idx+num_line_cells,line_idx:line_idx+num_line_cells] =1
        intra_class_intra_cell_blkdiag = np.triu(intra_class_intra_cell_blkdiag,k=1)
        
        inter_class_blkdiag = np.zeros((num_all_cells,num_all_cells))
        inter_class_blkdiag[line_idx:line_idx+num_line_cells,:]=1
        inter_class_blkdiag[:,line_idx:line_idx+num_line_cells] = 0
        
       
        masked_intra_conductance = np.multiply(avg_dist_mat,intra_class_blkdiag).flatten()
        intra_class_intra_cell_conductance = masked_intra_conductance[masked_intra_conductance!=0]
        
        masked_intra_class_inter_cell_conductance = np.multiply(avg_dist_mat,intra_class_intra_cell_blkdiag).flatten()
        intra_class_inter_cell_conductance = masked_intra_class_inter_cell_conductance[masked_intra_class_inter_cell_conductance!=0]
        
        inter_class_conductance = np.multiply(avg_dist_mat,inter_class_blkdiag).flatten()
        inter_class_inter_cell_conductance = inter_class_conductance[inter_class_conductance!=0]
        
        _,p_less_intra = mannwhitneyu(intra_class_intra_cell_conductance,intra_class_inter_cell_conductance,
                                 alternative='less')
        
        _,p_less_inter = mannwhitneyu(intra_class_inter_cell_conductance,inter_class_inter_cell_conductance,
                                 alternative='less')
        
        sns.set(font_scale=1.2,style='whitegrid')
        fig_d,ax_d = plt.subplots(figsize=(4,3))
        sns.distplot(intra_class_intra_cell_conductance,norm_hist=True,ax=ax_d)
        sns.distplot(intra_class_inter_cell_conductance,norm_hist=True,ax=ax_d)
        sns.distplot(inter_class_inter_cell_conductance,norm_hist=True,ax=ax_d)
        
        ax_d = draw_significance(intra_class_intra_cell_conductance, intra_class_inter_cell_conductance,
                                 p_less_intra/num_comparisons,ax_d,height_offset = .05)
        ax_d = draw_significance(intra_class_inter_cell_conductance, inter_class_inter_cell_conductance,
                                 p_less_inter/num_comparisons,ax_d,height_offset = .05)
        ax_d.grid(False)
        sns.despine(ax=ax_d)
        fig_d.savefig('figures/%s_dist_comp.pdf'%line,bbox_inches='tight')
        plt.close(fig_d)
        
        ax.add_patch(PathPatch(Path([[current_idx, current_idx],
                                     [current_idx+line_hof_models, current_idx],
                                     [current_idx+line_hof_models, current_idx+line_hof_models],
                                     [current_idx, current_idx]]),
                                     facecolor="none",edgecolor='k',
                                      linewidth=2))
        current_idx += line_hof_models
        line_idx += num_line_cells
    
    
    current_idx = 0
    for _ in range(num_all_cells):
        ax.add_patch(PathPatch(Path([[current_idx, current_idx],
                                      [current_idx+hof_num, current_idx],
                                      [current_idx+hof_num, current_idx+hof_num],
                                      [current_idx,current_idx]]),
                                      facecolor="none",
                                      edgecolor='k',
                                      linewidth=2))
        current_idx += hof_num
    
    g.fig.savefig('figures/inh_conductance_heatmap.pdf',bbox_inches='tight',dpi=200)
    g.fig.set_size_inches(5,5)
    plt.close(g.fig)