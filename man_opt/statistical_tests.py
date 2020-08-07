from itertools import combinations
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import mannwhitneyu
import man_opt.utils as man_utils
import pandas as pd


def pairwise_comp(data, cty_prop, prop_list, params, sig_level=0.05):
    """
    Pairwise comparison of parameters between cell-types 
    """

    diff_param_list = []
    p_val_list = []

    for param in params:
        for comb in combinations(prop_list, 2):
            cty_x, cty_y = comb
            paramx = data.loc[data[cty_prop] == cty_x, param].values
            paramy = data.loc[data[cty_prop] == cty_y, param].values
            _, p_val_x = mannwhitneyu(paramx, paramy, alternative='less')
            _, p_val_y = mannwhitneyu(paramy, paramx, alternative='less')
            comp_type = '%s<%s' % (
                cty_x, cty_y) if p_val_x < p_val_y else '%s<%s' % (cty_y, cty_x)
            p_val = min(p_val_x, p_val_y)
            sig_dict = {'Comp_type': comp_type,
                        'param': param}
            diff_param_list.append(sig_dict)
            p_val_list.append(p_val)

    # FDR correction for multiple comparison
    _, p_val_corrected = fdrcorrection(p_val_list)

    diff_param_df = pd.DataFrame(diff_param_list)
    diff_param_df['p_val'] = p_val_corrected
    diff_param_df['sig_level'] = diff_param_df['p_val'].apply(
        lambda x: man_utils.pval_to_sig(x))

    return diff_param_df
