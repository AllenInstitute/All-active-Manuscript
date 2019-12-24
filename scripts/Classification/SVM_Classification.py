import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ateamopt.utils import utility
from ateamopt.analysis.allactive_classification import Allactive_Classification as aa_clf
import os
import numpy as np
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix,accuracy_score  
from sklearn.model_selection import StratifiedKFold,GridSearchCV,\
            RandomizedSearchCV,train_test_split  
from sklearn.utils.multiclass import unique_labels
import scipy.stats as stats     

def SVM_classifier(X_df,y_df,feature_fields,target_field,hyp_search='random'):
        np.random.seed(0)
        svm_pipeline =  Pipeline([('scaler', StandardScaler()),
                                      ('svc', SVC(kernel='rbf'))])
        le = preprocessing.LabelEncoder()  
        y_df['label_encoder']= le.fit_transform(y_df[target_field])
        
        X_data = X_df.loc[:,feature_fields].values
        y_data = y_df['label_encoder'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
                X_data, y_data, test_size=0.3, stratify=y_data, random_state=0)
        
        ## Hyperparameter selection
        n_tune_split = 3
        
        if hyp_search == 'grid':
        
            ## Grid Search
            param_grid = {'svc__C':np.logspace(-2,3,100),'svc__gamma':np.logspace(-4,0,10)}
            tuned_pipeline = GridSearchCV(svm_pipeline, param_grid,
                                              cv=n_tune_split,n_jobs=4).fit(X_train, y_train)

        else:
            ## Randomized Search
            n_iter = 1000
            param_distrib = {'svc__C':stats.loguniform(1e-2,1000),
                             'svc__gamma':stats.loguniform(1e-4,1)}
            
            tuned_pipeline = RandomizedSearchCV(svm_pipeline, param_distrib, 
                        cv=n_tune_split,n_iter=n_iter,random_state=0,n_jobs=4).fit(X_train, y_train)
        
        tuned_C, tuned_gamma = (tuned_pipeline.best_params_['svc__C'],
                                    tuned_pipeline.best_params_['svc__gamma'])

        print('Tuned hyperparameters : C = %s, gamma = %s\n'%(tuned_C, tuned_gamma))
            
        y_pred_test = tuned_pipeline.predict(X_test)
        y_pred_chance = np.random.choice(y_test,len(y_test))
        confusion_matrix_svm = confusion_matrix(y_test, y_pred_test)
        
        score = accuracy_score(y_test, y_pred_test)
        chance_score = accuracy_score(y_test, y_pred_chance)
        
        classes = le.inverse_transform(unique_labels(y_test, \
                                        y_pred_test))
        
        df_conf_svm = pd.DataFrame(confusion_matrix_svm, classes,
              classes)
        df_conf_svm=df_conf_svm.div(df_conf_svm.sum(axis=1),axis=0) # Fraction of the actual no.s
        
        df_conf_svm *= 100
        delta_chance = score - chance_score  
        score = np.round(100*score,1)
        delta_chance = np.round(delta_chance*100,1)
        
        best_y_pred,best_y = le.inverse_transform(y_pred_test),\
                                    le.inverse_transform(y_test)
        
        return score,df_conf_svm,delta_chance,best_y,best_y_pred


data_path = os.path.join(os.getcwd(),os.pardir,os.pardir,'assets','aggregated_data')
mouse_data_filename = os.path.join(data_path,'Mouse_class_data.csv')
mouse_datatype_filename = os.path.join(data_path,'Mouse_class_datatype.csv')

morph_data_filename = os.path.join(data_path,'morph_data.csv')
morph_datatype_filename = os.path.join(data_path,'morph_datatype.csv')

sdk_data_filename = os.path.join(data_path,'sdk.csv')
sdk_datatype_filename = os.path.join(data_path,'sdk_datatype.csv')

param_data_filename = os.path.join(data_path,'allactive_params.csv')
param_datatype_filename = os.path.join(data_path,'allactive_params_datatype.csv')

train_ephys_max_amp_fname = os.path.join(data_path,'train_ephys_max_amp.csv')
train_ephys_max_amp_dtype_fname = os.path.join(data_path,'train_ephys_max_amp_dtype.csv')
train_ephys_max_amp_fields_fname = os.path.join(data_path,'train_ephys_max_amp_fields.json')

cre_coloring_filename = os.path.join(data_path,'rnaseq_sorted_cre.pkl')

clf_handler = aa_clf()
mouse_data_df = clf_handler.read_class_data(mouse_data_filename,mouse_datatype_filename)
morph_data = clf_handler.read_class_data(morph_data_filename,morph_datatype_filename)
morph_fields = clf_handler.get_data_fields(morph_data)
allensdk_data = clf_handler.read_class_data(mouse_data_filename,mouse_datatype_filename)
allensdk_fields = clf_handler.get_data_fields(allensdk_data)
ephys_data = clf_handler.read_class_data(train_ephys_max_amp_fname,
                                                  train_ephys_max_amp_dtype_fname)
ephys_fields = utility.load_json(train_ephys_max_amp_fields_fname)
hof_param_data = clf_handler.read_class_data(param_data_filename,param_datatype_filename)

cre_cluster = mouse_data_df.loc[mouse_data_df.hof_index==0,['Cell_id','Cre_line']]
bcre_cluster = mouse_data_df.loc[mouse_data_df.hof_index==0,['Cell_id','Broad_Cre_line']]

cre_color_dict = utility.load_pickle(cre_coloring_filename)
rna_seq_crelines = list(cre_color_dict.keys())
all_crelines = cre_cluster.Cre_line.unique().tolist()
cre_pal_all = {cre_:(mpl.colors.to_hex(cre_color_dict[cre_]) 
        if cre_ in rna_seq_crelines else 
             mpl.colors.to_hex('k')) for cre_ in all_crelines}

me_fields= ephys_fields + morph_fields
me_data = pd.merge(ephys_data,morph_data, how='left',on='Cell_id')
param_data = hof_param_data.loc[hof_param_data.hof_index == 0,]
param_data = param_data.drop(labels='hof_index',axis=1)
p_fields = list(param_data)
mp_fields= morph_fields+p_fields
mp_data = pd.merge(morph_data,param_data, how='left',on='Cell_id')


target_field_list = ['Broad_Cre_line','Cre_line']
feature_field_list = [ephys_fields,me_fields,p_fields,mp_fields]
feature_name_list = ['Ephys','Morph+Ephys','Model Parameters','Morph+Parameters']
feature_data_list = [ephys_data,me_data,param_data,mp_data]
cluster_data_list = [bcre_cluster,cre_cluster]

svm_scores_arr = np.zeros((len(target_field_list),len(feature_field_list)))
svm_conf_mat_grid = {}
best_pred_record = {}
delta_chance_arr = np.zeros_like(svm_scores_arr)
least_pop_index = 6

for ii,target_ in enumerate(target_field_list):
    for jj,feature_ in enumerate(feature_field_list):
        
        feature_fields = [feature_field_ for feature_field_ in feature_ \
                  if feature_field_ != 'Cell_id']
        
        feature_data = feature_data_list[jj]
        cluster_data = cluster_data_list[ii]
        df_clf = pd.merge(feature_data,cluster_data,how='left',
                        on='Cell_id')
        
        if target_ == 'Cre_line':
            df_clf = df_clf.loc[df_clf[target_].isin(rna_seq_crelines),]
            
        X_df,y_df,revised_features = clf_handler.prepare_data_clf\
            (df_clf,feature_fields,target_,
            least_pop=least_pop_index)
        
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print('Classifier Features : %s'%feature_name_list[jj])
        print('Classifier Target : %s'%target_)
        
        score,conf_df,delta_chance,best_y,best_y_pred= SVM_classifier(X_df,\
                 y_df,revised_features,target_,hyp_search='grid')
        
        if target_ == 'Broad_Cre_line':
            bcre_index_order = ['Htr3a','Sst','Pvalb','Pyr']
            conf_df = conf_df.reindex(index=bcre_index_order,columns=bcre_index_order)

        elif target_ == 'Cre_line':
            cre_indices = conf_df.index
            sorted_cre_indices = [cre_ind_ for cre_ind_ in rna_seq_crelines \
                                  if cre_ind_ in cre_indices]
            conf_df = conf_df.reindex(index=sorted_cre_indices,columns=
                                      sorted_cre_indices)
            cre_indices = [cre.split('-')[0] for cre in sorted_cre_indices]
            conf_df=pd.DataFrame(conf_df.values,index=cre_indices,columns=cre_indices)

        svm_scores_arr[ii,jj] = score
        svm_conf_mat_grid[ii,jj] = conf_df
        delta_chance_arr[ii,jj] = delta_chance    
        best_pred_record[ii,jj] = pd.DataFrame({'true':best_y,'predicted':best_y_pred})
    
print(svm_scores_arr)