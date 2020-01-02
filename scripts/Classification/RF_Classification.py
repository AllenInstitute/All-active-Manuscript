import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ateamopt.utils import utility
from ateamopt.analysis.allactive_classification import Allactive_Classification as aa_clf
import os
import numpy as np
import matplotlib as mpl
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix,accuracy_score  
from sklearn.model_selection import GridSearchCV,train_test_split  
from sklearn.utils.multiclass import unique_labels

def RF_classifier(X_df,y_df,feature_fields,target_field):
    np.random.seed(0)
    clf = RandomForestClassifier(random_state=0)
    
    le = preprocessing.LabelEncoder()   
    y_df['label_encoder']= le.fit_transform(y_df[target_field])
    
    X_data = X_df.loc[:,feature_fields].values
    y_data = y_df['label_encoder'].values

    X_train, X_test, y_train, y_test = train_test_split(
                X_data, y_data, test_size=0.3, stratify=y_data, random_state=0)
    
    
    ## Hyperparameter selection
    n_tune_split = 3
    
    ## Grid Search
    param_grid = {'n_estimators':np.arange(25,550,25),'max_leaf_nodes':np.arange(2,42,2)}
    tuned_clf = GridSearchCV(clf, param_grid,cv=n_tune_split,n_jobs=6).fit(X_train, y_train)
    
    tuned_max_trees, tuned_max_leaf_nodes = (tuned_clf.best_params_['n_estimators'],
                                    tuned_clf.best_params_['max_leaf_nodes'])

    print('Tuned hyperparameters : n_estimators = %s, max_leaf_nodes = %s\n'%(tuned_max_trees, 
                                  tuned_max_leaf_nodes))
    
    y_pred_test = tuned_clf.predict(X_test)
    y_pred_chance = np.random.choice(y_test,len(y_test))
    confusion_matrix_rf= confusion_matrix(y_test, y_pred_test)
    
    score = accuracy_score(y_test, y_pred_test)
    chance_score = accuracy_score(y_test, y_pred_chance)
    
    classes = le.inverse_transform(unique_labels(y_test, \
                                    y_pred_test))
    
    df_conf_rf = pd.DataFrame(confusion_matrix_rf, classes,classes)
    df_conf_rf =df_conf_rf.div(df_conf_rf.sum(axis=1),axis=0) # Fraction of the actual no.s
    
    df_conf_rf *= 100
    delta_chance = score - chance_score  
    score = np.round(100*score,1)
    delta_chance = np.round(delta_chance*100,1)
    
    best_y_pred,best_y = le.inverse_transform(y_pred_test),\
                                le.inverse_transform(y_test)
    
    feature_imp = pd.Series(tuned_clf.best_estimator_.feature_importances_,
                    index=feature_fields).sort_values(ascending=False)
    feature_fields_sorted = feature_imp.index.values
    feature_dict = {'importance': [], 'param_name' : []}
    for tree in tuned_clf.best_estimator_.estimators_:
        for i, param_name_ in enumerate(feature_fields_sorted):
            sorted_idx = np.where(np.array(feature_fields) == param_name_)[0][0]
            feature_dict['importance'].append(tree.feature_importances_[sorted_idx])
            feature_dict['param_name'].append(param_name_)
    feature_imp_df = pd.DataFrame(feature_dict)
     
    param_group_dict = feature_imp_df.groupby('param_name')['importance'].\
            agg(np.median).to_dict()
    params_sorted =  sorted(param_group_dict, key=param_group_dict.get,
                            reverse=True)
    return score,df_conf_rf,delta_chance,best_y,best_y_pred,feature_imp_df,params_sorted

data_path = os.path.join(os.getcwd(),os.pardir,os.pardir,'assets','aggregated_data')
mouse_data_filename = os.path.join(data_path,'Mouse_class_data.csv')
mouse_datatype_filename = os.path.join(data_path,'Mouse_class_datatype.csv')

param_data_filename = os.path.join(data_path,'allactive_params.csv')
param_datatype_filename = os.path.join(data_path,'allactive_params_datatype.csv')

me_cluster_data_filename = os.path.join(data_path,'tsne_mouse.csv')


cre_coloring_filename = os.path.join(data_path,'rnaseq_sorted_cre.pkl')

clf_handler = aa_clf()
mouse_data_df = clf_handler.read_class_data(mouse_data_filename,mouse_datatype_filename)
hof_param_data = clf_handler.read_class_data(param_data_filename,param_datatype_filename)
me_cluster_data = pd.read_csv(me_cluster_data_filename,index_col=None)
me_cluster_data.rename(columns={'specimen_id':'Cell_id'},inplace=True)
me_cluster_data['Cell_id'] = me_cluster_data['Cell_id'].astype(str)

param_data = hof_param_data.loc[hof_param_data.hof_index == 0,]
param_data = param_data.drop(labels='hof_index',axis=1)
ephys_cluster = me_cluster_data.loc[:,['Cell_id','ephys_cluster']]
me_cluster = me_cluster_data.loc[:,['Cell_id','me_type']]
cre_cluster = mouse_data_df.loc[mouse_data_df.hof_index==0,['Cell_id','Cre_line']]
bcre_cluster = mouse_data_df.loc[mouse_data_df.hof_index==0,['Cell_id','Broad_Cre_line']]

cre_color_dict = utility.load_pickle(cre_coloring_filename)
rna_seq_crelines = list(cre_color_dict.keys())
all_crelines = cre_cluster.Cre_line.unique().tolist()
cre_pal_all = {cre_:(mpl.colors.to_hex(cre_color_dict[cre_]) 
        if cre_ in rna_seq_crelines else 
             mpl.colors.to_hex('k')) for cre_ in all_crelines}

target_field_list = ['ephys_cluster','me_type','Broad_Cre_line','Cre_line']
pass_param_list= ['cm','e_pas', 'Ra','g_pas','gbar_Ih']

p_fields = list(param_data)
other_param_list = [p_field_ for p_field_ in p_fields 
                if p_field_ != 'Cell_id' and not p_field_.startswith(tuple(pass_param_list))]
feature_field_list  = [p_fields,p_fields]
feature_data_list = [param_data,param_data]
cluster_data_list = [ephys_cluster,me_cluster,bcre_cluster,cre_cluster]

imp_df_list = []
rf_scores_arr = np.zeros((len(feature_field_list),len(target_field_list)))
rf_conf_mat_grid = {}
best_pred_record = {}
rf_imp_dict,rf_sorted_params = {},{}
delta_chance_arr = np.zeros_like(rf_scores_arr)
least_pop_index = 6

for jj,target_ in enumerate(target_field_list):
    for ii,feature_ in enumerate(feature_field_list):
        
        if ii == 1:
            feature_fields = [feature_field_ for feature_field_ in feature_ \
                  if feature_field_ != 'Cell_id']
        else: # Only training on passive + Ih
            feature_fields = [feature_field_ for feature_field_ in feature_ \
                  if feature_field_ not in ['Cell_id']+other_param_list]
        
        feature_data = feature_data_list[ii]
        cluster_data = cluster_data_list[jj]
        df_clf = pd.merge(feature_data,cluster_data,how='left',
                        on='Cell_id')
        
        if target_ == 'Cre_line':
            df_clf = df_clf.loc[df_clf[target_].isin(rna_seq_crelines),]
            
        X_df,y_df,revised_features = clf_handler.prepare_data_clf\
            (df_clf,feature_fields,target_,
            least_pop=least_pop_index)
            
        score,conf_df,delta_chance,best_y,best_y_pred,df_imp,sorted_param_imp= \
                 RF_classifier(X_df,y_df,revised_features,target_)
                 
        if ii == 1:
            df_imp['Features'] = 'All'
        else:
            df_imp['Features'] = 'Passive+Ih'
        
        if target_ == 'Broad_Cre_line':
            bcre_index_order = ['Htr3a','Sst','Pvalb','Pyr']
            conf_df = conf_df.reindex(index=bcre_index_order,columns=bcre_index_order)
            df_imp['Classifier_target'] = 'Broad Cre-line'
        elif target_ == 'Cre_line':
            cre_indices = conf_df.index
            sorted_cre_indices = [cre_ind_ for cre_ind_ in rna_seq_crelines \
                                  if cre_ind_ in cre_indices]
            conf_df = conf_df.reindex(index=sorted_cre_indices,columns=
                                      sorted_cre_indices)
            cre_indices = [cre.split('-')[0] for cre in sorted_cre_indices]
            conf_df=pd.DataFrame(conf_df.values,index=cre_indices,columns=cre_indices)
            df_imp['Classifier_target'] = 'Cre-line'
        elif target_ == 'ephys_cluster':
            df_imp['Classifier_target'] = 'E-type'
        elif target_ == 'me_type':
            df_imp['Classifier_target'] = 'ME-type'
            
        rf_scores_arr[ii,jj] = score
        rf_conf_mat_grid[ii,jj] = conf_df
        delta_chance_arr[ii,jj] = delta_chance    
        rf_imp_dict[ii,jj] = df_imp
        rf_sorted_params[ii,jj] = rf_sorted_params
        best_pred_record[ii,jj] = pd.DataFrame({'true':best_y,'predicted':best_y_pred})

        imp_df_list.append(df_imp)

print(rf_scores_arr)