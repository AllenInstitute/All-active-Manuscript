from sklearn import metrics
import numpy as np
from man_opt.utils import bounding_box

def silhouette_score(estimator, X):
    cluster_labels = estimator.fit_predict(X)
    num_labels = len(set(cluster_labels))
    num_samples = X.shape[0]
    if num_labels == 1 or num_labels == num_samples:
        return -1
    else:
        return metrics.silhouette_score(X, cluster_labels)

def gap_statistic(estimator, X):
    np.random.seed(0)
    (xmin,xmax), (ymin,ymax) = bounding_box(X)
    n_references=1
    ref_inertia_rep = []
    for _ in range(n_references):
        ref_data = np.random.uniform(low=(xmin,ymin),high=(xmax,ymax),
                                     size=X.shape)
        estimator.fit(ref_data)
        ref_inertia_rep.append(estimator.inertia_)
    ref_inertia = np.mean(np.log(ref_inertia_rep))
    ref_std = np.std(np.log(ref_inertia_rep))
    estimator.fit(X)
    local_inertia = estimator.inertia_
    gap = ref_inertia-np.log(local_inertia)
    sk = ref_std*np.sqrt(1+1/n_references)
    return gap,sk    