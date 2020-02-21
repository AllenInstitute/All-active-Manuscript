from man_opt.analysis_tree_helpers import HTree
import matplotlib.pyplot as plt

#Load a tree from the .csv file
VISp_tree_filename = 'tree_VISp.csv'

htree = HTree(htree_file=VISp_tree_filename)
gluta_subtree = htree.get_subtree('n6')
fig = gluta_subtree.plot((10,4),fontsize=14,txtleafonly=True)
fig.savefig('figures/exc_subtree.svg',bbox_inches='tight')
plt.close(fig)