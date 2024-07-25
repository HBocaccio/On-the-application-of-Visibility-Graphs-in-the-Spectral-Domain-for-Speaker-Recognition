

import numpy as np
import os
import pandas as pd



# # # Define root directories
root_dir = ''
variables_dir = os.path.join(root_dir, 'variables', 'vg_metrics')
target_dir = os.path.join(root_dir, 'variables', 'models_thresholds')
if not os.path.exists(target_dir):
    os.makedirs(target_dir)



# # # Define root variables
vocalization_speakers = np.loadtxt(os.path.join(variables_dir, 'vocalization_speakers.txt'), dtype='str')
vocalization_vowels = np.loadtxt(os.path.join(variables_dir, 'vocalization_vowels.txt'), dtype='str')
vocalization_files = np.loadtxt(os.path.join(variables_dir, 'vocalization_files.txt'), dtype='str')
subject_names = [s for s in np.unique(vocalization_speakers)]
vowels = ['a', 'e', 'i', 'o', 'u']
M = 10
seed = 42
thresholds = np.arange(0.5, 0.95+0.01, 0.05)
N_features = 1000
orders = [13]



# # # Split data
for o in np.arange(len(orders)):
    order = orders[o]
    dir_order = os.path.join(target_dir, 'order_'+str(order))
    if not(os.path.isdir(os.path.join(dir_order))):
        os.mkdir(os.path.join(dir_order))
    variables_dir_order = os.path.join(variables_dir, 'order_'+str(order))
    df_metrics = pd.read_excel(os.path.join(variables_dir_order, 'df_metrics.xlsx'), sheet_name='df_metrics')
    for t in np.arange(len(thresholds)):
        th = round(thresholds[t], 2)
        dir_threshold = os.path.join(dir_order, 'th_'+str(round(th, 2)))
        if not(os.path.isdir(os.path.join(dir_threshold))):
            os.mkdir(os.path.join(dir_threshold))
        variables_dir_threshold = os.path.join(variables_dir_order, 'th_'+str(round(th, 2)))
        giant_indices = np.loadtxt(os.path.join(variables_dir_threshold, 'giant_indices.txt')).astype('int')
        df_metrics_subset = df_metrics.iloc[giant_indices].loc[df_metrics['Speaker'].isin(subject_names)]
        np.random.seed(seed)
        # p_train = 0.5
        # p_test = 1 - p_train
        p_train = 0.4
        p_val = 0.3
        p_test = 1 - p_train - p_val
        for m in np.arange(M):
            model_dir = os.path.join(dir_threshold, 'model_'+str(m+1).zfill(2))
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            sfiles = []
            sfiles_train = []
            sfiles_val = []
            sfiles_test = []
            a = []
            n_files_train = np.zeros((len(subject_names), len(vowels)))
            n_files_val = np.zeros((len(subject_names), len(vowels)))
            n_files_test = np.zeros((len(subject_names), len(vowels)))
            for s in np.arange(len(subject_names)):
                afiles = []
                afiles_test = []
                afiles_train = []
                afiles_val = []
                for v in np.arange(len(vowels)):
                    df_tmp = df_metrics_subset[(df_metrics_subset['Speaker'] == subject_names[s])&
                                               (df_metrics_subset['Vowel'] == vowels[v])]
                    afiles_tmp = list(df_tmp['File'].ravel())
                    n_train = int(np.floor(len(afiles_tmp)*p_train))
                    n_test = int(np.floor(len(afiles_tmp)*p_test))
                    afiles_test_tmp = list(np.random.choice(afiles_tmp, size=n_test, replace=False))
                    afiles_dev_tmp = [a for a in afiles_tmp if a not in afiles_test_tmp]
                    afiles_train_tmp = list(np.random.choice(afiles_dev_tmp, size=n_train, replace=False))
                    afiles_val_tmp = [a for a in afiles_dev_tmp if a not in afiles_train_tmp]
                    n_val = int(len(afiles_val_tmp))
                    afiles += afiles_tmp
                    afiles_test += afiles_test_tmp
                    afiles_train += afiles_train_tmp
                    afiles_val += afiles_val_tmp
                    n_files_train[s, v] = n_train
                    n_files_val[s, v] = n_val
                    n_files_test[s, v] = n_test
                sfiles.append(afiles)
                sfiles_train.append(afiles_train)
                sfiles_val.append(afiles_val)
                sfiles_test.append(afiles_test)
                a.append(int(subject_names[s].split('S')[1]))
            df_train = pd.DataFrame({'Subject': a,
                                     'Files': sfiles,
                                     'Files_train': sfiles_train,
                                     'Files_val': sfiles_val,
                                     'Files_test': sfiles_test})
            df_train.to_csv(os.path.join(model_dir,'df_train.csv'), index=False)
            np.savetxt(os.path.join(model_dir, 'n_files_train.txt'), n_files_train, fmt='%i')
            np.savetxt(os.path.join(model_dir, 'n_files_val.txt'), n_files_val, fmt='%i')
            np.savetxt(os.path.join(model_dir, 'n_files_test.txt'), n_files_test, fmt='%i')



# # # Get vowels features combination indices
for o in np.arange(len(orders)):
    order = orders[o]
    dir_order = os.path.join(target_dir, 'order_'+str(order))
    variables_dir_order = os.path.join(variables_dir, 'order_'+str(order))
    df_metrics = pd.read_excel(os.path.join(variables_dir_order, 'df_metrics.xlsx'), sheet_name='df_metrics')
    for t in np.arange(len(thresholds)):
        th = round(thresholds[t], 2)
        dir_threshold = os.path.join(dir_order, 'th_'+str(round(th, 2)))
        if not(os.path.isdir(os.path.join(dir_threshold))):
            os.mkdir(os.path.join(dir_threshold))
        variables_dir_threshold = os.path.join(variables_dir_order, 'th_'+str(round(th, 2)))
        giant_indices = np.loadtxt(os.path.join(variables_dir_threshold, 'giant_indices.txt')).astype('int')
        df_metrics_subset = df_metrics.iloc[giant_indices].loc[df_metrics['Speaker'].isin(subject_names)]
        np.random.seed(seed)
        for m in np.arange(M):
            model_dir = os.path.join(dir_threshold, 'model_'+str(m+1).zfill(2))
            df_train = pd.read_csv(os.path.join(model_dir, 'df_train.csv'))
            sfiles_train = [eval(a) for a in df_train['Files_train']]
            sfiles_val = [eval(a) for a in df_train['Files_val']]
            sfiles_test = [eval(a) for a in df_train['Files_test']]
            indices_train = np.zeros((len(subject_names)*N_features, len(vowels))).astype('int')
            indices_val = np.zeros((len(subject_names)*N_features, len(vowels))).astype('int')
            indices_test = np.zeros((len(subject_names)*N_features, len(vowels))).astype('int')
            e = 0
            for s in np.arange(len(subject_names)):
                df_tmp = df_metrics_subset[df_metrics_subset['Speaker'] == subject_names[s]]
                afiles = list(df_tmp['File'].ravel())
                afiles_train = sfiles_train[s]
                afiles_val = sfiles_val[s]
                afiles_test = sfiles_test[s]
                for n in np.arange(N_features):
                    for v in np.arange(len(vowels)):
                        vfiles_train = [f for f in sfiles_train[s] if vowels[v] in f[:-4]]
                        vfiles_val = [f for f in sfiles_val[s] if vowels[v] in f[:-4]]
                        vfiles_test = [f for f in sfiles_test[s] if vowels[v] in f[:-4]]
                        indices_train[e, v] = np.random.randint(len(vfiles_train))
                        indices_val[e, v] = np.random.randint(len(vfiles_val))
                        indices_test[e, v] = np.random.randint(len(vfiles_test))
                    e = e + 1
            np.savetxt(os.path.join(model_dir, 'indices_train.txt'), indices_train, fmt='%i')
            np.savetxt(os.path.join(model_dir, 'indices_val.txt'), indices_val, fmt='%i')
            np.savetxt(os.path.join(model_dir, 'indices_test.txt'), indices_test, fmt='%i')


