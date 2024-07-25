

import numpy as np
import os
import pandas as pd
import bct



# # # Define root directories
root_dir = ''
variables_dir = os.path.join(root_dir, 'variables')
target_dir = os.path.join(variables_dir, 'vg_metrics')



# # # Define root variables
vocalization_speakers = np.loadtxt(os.path.join(target_dir, 'vocalization_speakers.txt'), dtype='str')
vocalization_vowels = np.loadtxt(os.path.join(target_dir, 'vocalization_vowels.txt'), dtype='str')
vocalization_files = np.loadtxt(os.path.join(target_dir, 'vocalization_files.txt'), dtype='str')
subject_names = [s for s in np.unique(vocalization_speakers)]
vowels = ['a', 'e', 'i', 'o', 'u']
sr = 11025
worN = 512
orders = np.arange(10,21)



# # # Get binarization
th = 0.9
for o in np.arange(len(orders)):
    order = orders[o]
    dir_order = os.path.join(target_dir, 'order_'+str(order))
    H_functions = np.loadtxt(os.path.join(dir_order, 'H_functions.txt'))

    node_links = []
    total_links = []
    comms = []
    for s in np.arange(len(subject_names)):
        for v in np.arange(len(vowels)):
            h_matrix = H_functions[:, (vocalization_speakers == subject_names[s]) &
                                   (vocalization_vowels == vowels[v])]
            h_corr = np.corrcoef(np.log10(h_matrix**2).T)
            h_corr[np.diag_indices_from(h_corr)] = 0
            node_links.append(np.sum((h_corr > th).astype('int'), axis=1))
            total_links.append(np.array([h_corr.shape[0]-1]*h_corr.shape[0]))
            h_corr_binary = (h_corr > th).astype('int')
            comms.append(bct.modularity_und(h_corr_binary)[0])
    node_links = np.hstack(node_links)
    total_links = np.hstack(total_links)
    df_communities = pd.DataFrame({'Speaker': vocalization_speakers,
                                   'Vowel': vocalization_vowels,
                                   'File': vocalization_files,
                                   'Cluster': np.hstack(comms),
                                   'Degree': node_links,
                                   'Links': total_links,
                                   })
    df_communities.to_excel(os.path.join(dir_order, 'df_communities.xlsx'))



# # # Get giant component indices
for o in np.arange(len(orders)):
    order = orders[o]
    dir_order = os.path.join(target_dir, 'order_'+str(order))
    df_communities = pd.read_excel(os.path.join(dir_order, 'df_communities.xlsx'))
    giant_indices = np.array([])
    giant_sizes = np.array([])
    for s in np.arange(len(subject_names)):
        for v in np.arange(len(vowels)):
            df_communities_tmp = df_communities[(df_communities['Speaker'] == subject_names[s]) &
                                                (df_communities['Vowel'] == vowels[v])]
            cluster_sizes = df_communities_tmp.groupby(['Cluster'])['File'].count()
            cluster_giant = np.array(cluster_sizes.index)[np.argmax(cluster_sizes)]
            giant_indices_tmp = np.array(df_communities_tmp[df_communities_tmp['Cluster'] == cluster_giant].index)
            giant_indices = np.append(giant_indices, giant_indices_tmp).astype('int')
            giant_sizes_tmp = np.array(np.max(cluster_sizes))
            giant_sizes = np.append(giant_sizes, giant_sizes_tmp).astype('int')
    np.savetxt(os.path.join(dir_order, 'giant_indices.txt'), giant_indices, fmt='%i')
    np.savetxt(os.path.join(dir_order, 'giant_sizes.txt'), giant_sizes, fmt='%i')


