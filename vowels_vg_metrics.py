

import numpy as np
import os
# import scipy as sp
import pandas as pd
# import librosa
# import networkx as nx
import bct

# # # Non-nested functions
def nvg_dc(series, left=None, right=None, timeLine=None, all_visible = None):
    L = len(series)
    if left == None : left = 0
    if right == None : right = L
    if timeLine == None : timeLine = list(range(left, right))
    if all_visible == None : all_visible = []
    node_visible = []
    if left < right:
        k = list(series)[left:right].index(max(list(series[left:right]))) + left
        for i in list(range(left,right)):
            if i != k :
                a = min(i,k)
                b = max(i,k)
                ya = float(series[a])
                ta = timeLine[a]
                yb = float(series[b])
                tb = timeLine[b]
                yc = series[a+1:b]
                tc = timeLine[a+1:b]
                if all(yc[j] < (ya + (yb - ya)*(tc[j] - ta)/(tb-ta)) for j in list(range(len(yc)))):
                    node_visible.append(timeLine[i])
        if len(node_visible)>0 : all_visible.append([timeLine[k], node_visible])
        nvg_dc(series, left, k, timeLine, all_visible = all_visible)
        nvg_dc(series, k+1, right, timeLine, all_visible = all_visible)
    return all_visible

def get_nvg_dc(ts):
    out = nvg_dc(ts)
    Adj = np.zeros((ts.size, ts.size))
    for el in out:
        Adj[el[0], el[1]] = 1
        Adj[el[-1::-1][0], el[-1::-1][1]] = 1
    return Adj


# # # Define root directories
root_dir = ''
# data_dir = os.path.join(root_dir, 'data', 'vowels_data_wav_segments_curated')
variables_dir = os.path.join(root_dir, 'variables')
if not os.path.exists(variables_dir):
    os.makedirs(variables_dir)
target_dir = os.path.join(variables_dir, 'vg_metrics')
if not os.path.exists(target_dir):
    os.makedirs(target_dir)


# # # Define root variables
vocalization_speakers = np.loadtxt(os.path.join(target_dir, 'vocalization_speakers.txt'), dtype='str')
vocalization_vowels = np.loadtxt(os.path.join(target_dir, 'vocalization_vowels.txt'), dtype='str')
vocalization_files = np.loadtxt(os.path.join(target_dir, 'vocalization_files.txt'), dtype='str')
# subject_names = os.listdir(data_dir)
subject_names = [s for s in np.unique(vocalization_speakers)]
vowels = ['a', 'e', 'i', 'o', 'u']
sr = 11025
worN = 512
orders = np.arange(10,21)
# vocalization_speakers = []
# vocalization_vowels = []
# vocalization_files = []
# for s in np.arange(len(subject_names)):
#     for v in np.arange(len(vowels)):
#         vowel = vowels[v]
#         vowel_files = [filename for filename in os.listdir(os.path.join(data_dir, subject_names[s]))
#                        if filename.startswith(str(vowel))]
#         vocalization_speakers = vocalization_speakers + [subject_names[s]]*len(vowel_files)
#         vocalization_vowels = vocalization_vowels + [vowel]*len(vowel_files)
#         vocalization_files = vocalization_files + vowel_files
# vocalization_speakers = np.array(vocalization_speakers)
# vocalization_vowels = np.array(vocalization_vowels)
# vocalization_files = np.array(vocalization_files)
# np.savetxt(os.path.join(target_dir, 'vocalization_speakers.txt'), vocalization_speakers, fmt='%s')
# np.savetxt(os.path.join(target_dir, 'vocalization_vowels.txt'), vocalization_vowels, fmt='%s')
# np.savetxt(os.path.join(target_dir, 'vocalization_files.txt'), vocalization_files, fmt='%s')

# # # Get spectrum functions
# for o in np.arange(len(orders)):
#     order = orders[o]
#     dir_order = os.path.join(target_dir, 'order_'+str(order))
#     if not(os.path.isdir(os.path.join(dir_order))):
#         os.mkdir(os.path.join(dir_order))
#     H_functions = np.zeros((worN, vocalization_files.size))
#     freqs_tmp = np.zeros((worN, vocalization_files.size))
#     lpcs = np.zeros((order+1, vocalization_files.size))
#     for e in np.arange(vocalization_files.size):
#         speaker = vocalization_speakers[e]
#         vowel = vocalization_vowels[e]
#         file = vocalization_files[e]
#         y, sr = librosa.load(os.path.join(data_dir, speaker, file), sr=sr, dtype=np.float64)
#         d = librosa.lpc(y, order=order)
#         [f, h] = sp.signal.freqz(b=1, a=d, worN=worN, fs=sr)
#         H_functions[:, e] = np.abs(h)
#         freqs_tmp[:, e] = f
#         lpcs[:, e] = d
#     if sum([sum(freqs_tmp[:,i]-freqs_tmp[:,0]) for i in np.arange(freqs_tmp.shape[1])]) == 0:
#         freqs = np.hstack(np.unique(freqs_tmp, axis=1))
#     np.savetxt(os.path.join(dir_order, 'H_functions.txt'), H_functions)
#     np.savetxt(os.path.join(dir_order, 'freqs.txt'), freqs)
#     np.savetxt(os.path.join(dir_order, 'lpcs.txt'), lpcs)


# # # Get visibility graph metrics
for o in np.arange(len(orders)):
    order = orders[o]
    dir_order = os.path.join(target_dir, 'order_'+str(order))
    H_functions = np.loadtxt(os.path.join(dir_order, 'H_functions.txt'))
    degrees = np.zeros(H_functions.shape)
    avg_degrees = []
    diameters = []
    densities = []
    avg_shortest_path_lengths = []
    clustering_coefficients = []
    communities = np.zeros(H_functions.shape)
    Qs = []
    for i in np.arange(len(vocalization_files)):
        h = H_functions[:, i]
        Adj = get_nvg_dc(np.log10(h**2))
        degrees[:, i] = bct.degrees_und(Adj)
        avg_degrees.append(np.mean(degrees[:, i]))
        diameters.append(bct.charpath(bct.distance_bin(Adj))[4])
        densities.append(bct.density_und(Adj)[0])
        avg_shortest_path_lengths.append(bct.charpath(bct.distance_bin(Adj))[0])
        clustering_coefficients.append(bct.clustering_coef_bu(Adj).mean())
        comm, Q = bct.modularity_und(Adj)
        communities[:, i] = comm
        Qs.append(Q)
    np.savetxt(os.path.join(dir_order, 'degrees.txt'), degrees, fmt='%i')
    np.savetxt(os.path.join(dir_order, 'avg_degrees.txt'), avg_degrees)
    np.savetxt(os.path.join(dir_order, 'diameters.txt'), diameters)
    np.savetxt(os.path.join(dir_order, 'densities.txt'), densities)
    np.savetxt(os.path.join(dir_order, 'avg_shortest_path_lengths.txt'), avg_shortest_path_lengths)
    np.savetxt(os.path.join(dir_order, 'clustering_coefficients.txt'), clustering_coefficients)
    np.savetxt(os.path.join(dir_order, 'communities.txt'), communities, fmt='%i')
    np.savetxt(os.path.join(dir_order, 'Qs.txt'), Qs)
    df_metrics = pd.DataFrame({'Speaker': vocalization_speakers,
                               'Vowel': vocalization_vowels,
                               'File': vocalization_files,
                               'AvgDegree': avg_degrees,
                               'Diameter': diameters,
                               'Density': densities,
                               'ASPL': avg_shortest_path_lengths,
                               'CC': clustering_coefficients,
                               'Q': Qs,
                               })
    df_metrics.to_excel(os.path.join(dir_order, 'df_metrics.xlsx'), sheet_name='df_metrics')
#     with pd.ExcelWriter(os.path.join(dir_order,'df_metrics_tables.xlsx')) as writer:
#         for s in np.arange(len(subject_names)):
#             df_metrics[df_metrics['Speaker'] == subject_names[s]].to_excel(writer, sheet_name=str(subject_names[s]))

