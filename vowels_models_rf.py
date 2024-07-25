

import numpy as np
import os
import pandas as pd
from ast import literal_eval
import sklearn as sk
from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics



# # # Define root directories
root_dir = ''
variables_dir = os.path.join(root_dir, 'variables', 'vg_metrics')
target_dir = os.path.join(root_dir, 'variables', 'models')



# # # Define root variables
vocalization_speakers = np.loadtxt(os.path.join(variables_dir, 'vocalization_speakers.txt'), dtype='str')
vocalization_vowels = np.loadtxt(os.path.join(variables_dir, 'vocalization_vowels.txt'), dtype='str')
vocalization_files = np.loadtxt(os.path.join(variables_dir, 'vocalization_files.txt'), dtype='str')
subject_names = [s for s in np.unique(vocalization_speakers)]
vowels = ['a', 'e', 'i', 'o', 'u']
M = 10
seed = 42
orders = np.arange(10,21)
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
        indices_train = np.loadtxt(os.path.join(model_dir, 'indices_train.txt'), dtype ='int')
        indices_val = np.loadtxt(os.path.join(model_dir, 'indices_val.txt'), dtype ='int')
        indices_test = np.loadtxt(os.path.join(model_dir, 'indices_test.txt'), dtype ='int')
        y_subjects = sum([[s]*N_features for s in subject_names], [])
        subject2label_dict = dict(zip(subject_names, list(np.arange(len(subject_names)))))
        y = np.array([subject2label_dict[s] for s in y_subjects])
        y_train = y
        y_val = y
        y_test = y
        X_train = np.zeros((len(subject_names)*N_features, len(vowels)*df_metrics_subset.iloc[:,4:].shape[1]))
        X_val = np.zeros((len(subject_names)*N_features, len(vowels)*df_metrics_subset.iloc[:,4:].shape[1]))
        X_test = np.zeros((len(subject_names)*N_features, len(vowels)*df_metrics_subset.iloc[:,4:].shape[1]))
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
                    row_train = np.append(row_train, df_tmp[(df_tmp['Vowel'] == vowels[v])&
                                                            (df_tmp['File'] == vfile_train)].iloc[:,4:].values)
                    row_val = np.append(row_val, df_tmp[(df_tmp['Vowel'] == vowels[v])&
                                                        (df_tmp['File'] == vfile_val)].iloc[:,4:].values)
                    row_test = np.append(row_test, df_tmp[(df_tmp['Vowel'] == vowels[v])&
                                                          (df_tmp['File'] == vfile_test)].iloc[:,4:].values)
                X_train[e, :] = np.hstack(row_train)
                X_val[e, :] = np.hstack(row_val)
                X_test[e, :] = np.hstack(row_test)
                e = e + 1
        # # # Grid search
        X_train_combined = np.concatenate((X_train, X_val), axis=0)
        y_train_combined = np.concatenate((y_train, y_val), axis=0)
        np.random.seed(seed)
        test_fold = np.concatenate((np.full(X_train.shape[0], -1), np.zeros(X_val.shape[0])))
        ps = sk.model_selection.PredefinedSplit(test_fold=test_fold)
        param_grid = {'n_estimators':  np.arange(5, 50+1, 5),
                      'max_depth': np.arange(5, 15+1),
                     }
        model = sk.ensemble.RandomForestClassifier(random_state=seed, verbose=False)
        grid_search = sk.model_selection.GridSearchCV(estimator=model, param_grid=param_grid, cv=ps, n_jobs=-1,
                                                      refit=False, return_train_score=True)
        grid_search.fit(X_train_combined, y_train_combined);
        df_results = pd.DataFrame(grid_search.cv_results_)
        df_results = df_results.drop(['std_fit_time', 'std_score_time', 'split0_test_score',
                                      'std_test_score', 'split0_train_score', 'std_train_score'], axis=1)
        params = ['param_'+p for p in param_grid.keys()]
        df_results[params] = df_results[params].astype(np.float64)
        df_results.to_csv(os.path.join(model_dir, 'df_results_rf.csv'), index=False)
        # # # Train and test
        np.random.seed(seed)
        best_params = df_results['params'][np.argmax(df_results['mean_test_score'])]
        clf = sk.ensemble.RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                                 max_depth=best_params['max_depth'],
                                                 n_jobs=-1, random_state=seed)
        clf.fit(X_train_combined, y_train_combined)
        y_pred_test = clf.predict(X_test)
        importances_mean = clf.feature_importances_
        importances_std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
        estimators = np.array([tree.feature_importances_ for tree in clf.estimators_])
        np.savetxt(os.path.join(model_dir, 'y_test_rf.txt'), y_test, fmt='%i')
        np.savetxt(os.path.join(model_dir, 'y_pred_test_rf.txt'), y_pred_test, fmt='%i')
        np.savetxt(os.path.join(model_dir, 'importances_mean_rf.txt'), importances_mean)
        np.savetxt(os.path.join(model_dir, 'importances_std_rf.txt'), importances_std)
        np.savetxt(os.path.join(model_dir, 'estimators_rf.txt'), estimators)


