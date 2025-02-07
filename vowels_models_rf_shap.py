

import numpy as np
import os
import pandas as pd
from ast import literal_eval
import sklearn as sk
from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics
import shap

# # # Define root directories
root_dir = ''
# data_dir = os.path.join(root_dir, 'data', 'vowels_data_wav_segments_curated')
variables_dir = os.path.join(root_dir, 'variables', 'vg_metrics')
target_dir = os.path.join(root_dir, 'variables', 'models')


# # # Define root variables
vocalization_speakers = np.loadtxt(os.path.join(variables_dir, 'vocalization_speakers.txt'), dtype='str')
vocalization_vowels = np.loadtxt(os.path.join(variables_dir, 'vocalization_vowels.txt'), dtype='str')
vocalization_files = np.loadtxt(os.path.join(variables_dir, 'vocalization_files.txt'), dtype='str')
# subject_names = os.listdir(data_dir)
subject_names = [s for s in np.unique(vocalization_speakers)]
vowels = ['a', 'e', 'i', 'o', 'u']
M = 10
seed = 42  # 6 #5 #4 #2 #42
orders = np.arange(10, 21)
N_features = 1000
droped = ['AvgDegree', 'Diameter']

# # # Get vowels features combination matrices
for o in np.arange(len(orders)):
    order = orders[o]
    dir_order = os.path.join(target_dir, 'order_'+str(order))
    variables_dir_order = os.path.join(variables_dir, 'order_'+str(order))
    giant_indices = np.loadtxt(os.path.join(variables_dir_order, 'giant_indices.txt')).astype('int')
    df_metrics = pd.read_excel(os.path.join(variables_dir_order, 'df_metrics.xlsx'), sheet_name='df_metrics')
    df_metrics_subset = df_metrics.iloc[giant_indices].loc[df_metrics['Speaker'].isin(subject_names)].drop(droped, axis=1)
    np.random.seed(seed)
    for m in np.arange(M):
        model_dir = os.path.join(dir_order, 'model_'+str(m+1).zfill(2))
        df_train = pd.read_csv(os.path.join(model_dir, 'df_train.csv'))
        sfiles = [eval(a) for a in df_train['Files']]
        sfiles_train = [eval(a) for a in df_train['Files_train']]
        sfiles_val = [eval(a) for a in df_train['Files_val']]
        sfiles_test = [eval(a) for a in df_train['Files_test']]
        indices_train = np.loadtxt(os.path.join(model_dir, 'indices_train.txt'), dtype='int')
        indices_val = np.loadtxt(os.path.join(model_dir, 'indices_val.txt'), dtype='int')
        indices_test = np.loadtxt(os.path.join(model_dir, 'indices_test.txt'), dtype='int')
        y_subjects = sum([[s]*N_features for s in subject_names], [])
        subject2label_dict = dict(zip(subject_names, list(np.arange(len(subject_names)))))
        y = np.array([subject2label_dict[s] for s in y_subjects])
        y_train = y
        y_val = y
        y_test = y
        X_train = np.zeros((len(subject_names)*N_features, len(vowels)*df_metrics_subset.iloc[:, 4:].shape[1]))
        X_val = np.zeros((len(subject_names)*N_features, len(vowels)*df_metrics_subset.iloc[:, 4:].shape[1]))
        X_test = np.zeros((len(subject_names)*N_features, len(vowels)*df_metrics_subset.iloc[:, 4:].shape[1]))
        e = 0
        for s in np.arange(len(subject_names)):
            df_tmp = df_metrics_subset[df_metrics_subset['Speaker'] == subject_names[s]]
            afiles = sfiles[s]
            afiles_train = sfiles_train[s]
            afiles_val = sfiles_val[s]
            afiles_test = sfiles_test[s]
            for n in np.arange(N_features):
                row_train = []
                row_val = []
                row_test = []
                for v in np.arange(len(vowels)):
                    vfiles_train = [f for f in sfiles_train[s] if vowels[v] in f[:-4]]
                    vfiles_val = [f for f in sfiles_val[s] if vowels[v] in f[:-4]]
                    vfiles_test = [f for f in sfiles_test[s] if vowels[v] in f[:-4]]
                    vfile_train = vfiles_train[indices_train[e, v]]
                    vfile_val = vfiles_val[indices_val[e, v]]
                    vfile_test = vfiles_test[indices_test[e, v]]
                    row_train = np.append(row_train, df_tmp[(df_tmp['Vowel'] == vowels[v]) &
                                                            (df_tmp['File'] == vfile_train)].iloc[:, 4:].values)
                    row_val = np.append(row_val, df_tmp[(df_tmp['Vowel'] == vowels[v]) &
                                                        (df_tmp['File'] == vfile_val)].iloc[:, 4:].values)
                    row_test = np.append(row_test, df_tmp[(df_tmp['Vowel'] == vowels[v]) &
                                                          (df_tmp['File'] == vfile_test)].iloc[:, 4:].values)
                X_train[e, :] = np.hstack(row_train)
                X_val[e, :] = np.hstack(row_val)
                X_test[e, :] = np.hstack(row_test)
                e = e + 1
        X_train_combined = np.concatenate((X_train, X_val), axis=0)
        y_train_combined = np.concatenate((y_train, y_val), axis=0)
        # # # Train and test
        np.random.seed(seed)
        df_results = pd.read_csv(os.path.join(model_dir, 'df_results_rf.csv'), converters={'params': literal_eval})
        best_params = df_results['params'][np.argmax(df_results['mean_test_score'])]
        clf = sk.ensemble.RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                                 max_depth=best_params['max_depth'],
                                                 n_jobs=-1, random_state=seed)
        clf.fit(X_train_combined, y_train_combined)
        y_pred_test_shap = clf.predict(X_test)
        # y_pred_test = np.loadtxt(os.path.join(model_dir, 'y_pred_test_rf.txt'), dtype='int')
        # if not(np.all(y_pred_test == y_pred_test_shap)):
        explainer_tree = shap.TreeExplainer(clf)
        shap_values_train = explainer_tree.shap_values(X_train_combined)
        shap_values_test = explainer_tree.shap_values(X_test)
        shap_importance_train = np.abs(shap_values_train).mean(axis=(0, 2))
        shap_importance_test = np.abs(shap_values_test).mean(axis=(0, 2))
        importances_mean = np.loadtxt(os.path.join(model_dir, 'importances_mean_rf.txt'))

        explainer_kernel_train = shap.KernelExplainer(clf.predict, shap.sample(X_train_combined, 50))
        shap_values_train_kernel = explainer_kernel_train.shap_values(shap.sample(X_train_combined, 50))
        explainer_kernel_test = shap.KernelExplainer(clf.predict, shap.sample(X_test, 50))
        shap_values_test_kernel = explainer_kernel_test.shap_values(shap.sample(X_test, 50))
        shap_importance_train_kernel = np.abs(shap_values_train_kernel).mean(axis=0)
        shap_importance_test_kernel = np.abs(shap_values_test_kernel).mean(axis=0)

        np.savetxt(os.path.join(model_dir, 'y_pred_test_rf_shap.txt'), y_pred_test_shap, fmt='%i')
        np.savetxt(os.path.join(model_dir, 'shap_importance_train.txt'), shap_importance_train)
        np.savetxt(os.path.join(model_dir, 'shap_importance_test.txt'), shap_importance_test)
        np.savetxt(os.path.join(model_dir, 'shap_importance_train_kernel.txt'), shap_importance_train_kernel)
        np.savetxt(os.path.join(model_dir, 'shap_importance_test_kernel.txt'), shap_importance_test_kernel)
