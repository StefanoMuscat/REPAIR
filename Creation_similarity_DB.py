from math import log
from math import ceil
import os
import sys
import pandas as pd
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from rdkit.DataManip.Metric import GetTanimotoDistMat
from rdkit.DataManip.Metric import GetTanimotoSimMat
from rdkit import rdBase
from rdkit.Chem import PandasTools
import matplotlib.pyplot as plt


# Before running this script, you need to do the following things:
# -     download the database from the following link: https://coconut.naturalproducts.net/download/absolutesmiles
# -     Add the following header to the file: "SMILES ID"
# -     Rename the file to "COCONUT_DB.smi"

# Path file
filename = "COCONUT_DB.smi"
md_db = "medical_device_DB.smi"

def calculate_counters(data_sets, c_threshold=None, w_factor="fraction"):

    # Setting matches
    total_data = np.sum(data_sets, axis=0)
    n_fingerprints = int(total_data[-1])
    c_total = total_data[:-1]

    # Assign c_threshold
    if not c_threshold:
        c_threshold = n_fingerprints % 2
    if isinstance(c_threshold, str):
        if c_threshold != 'dissimilar':
            raise TypeError(
                "c_threshold must be None, 'dissimilar', or an integer.")
        else:
            c_threshold = ceil(n_fingerprints / 2)
    if isinstance(c_threshold, int):
        if c_threshold >= n_fingerprints:
            raise ValueError(
                "c_threshold cannot be equal or greater than n_fingerprints.")
        c_threshold = c_threshold

    # Set w_factor
    if w_factor:
        if "power" in w_factor:
            power = int(w_factor.split("_")[-1])

            def f_s(d):
                return power**-float(n_fingerprints - d)

            def f_d(d):
                return power**-float(d - n_fingerprints % 2)
        elif w_factor == "fraction":
            def f_s(d):
                return d/n_fingerprints

            def f_d(d):
                return 1 - (d - n_fingerprints % 2)/n_fingerprints
        else:
            def f_s(d):
                return 1

            def f_d(d):
                return 1
    else:
        def f_s(d):
            return 1

        def f_d(d):
            return 1

    # Calculate a, d, b + c
    a = 0
    w_a = 0
    d = 0
    w_d = 0
    total_dis = 0
    total_w_dis = 0
    for s in c_total:
        if 2 * s - n_fingerprints > c_threshold:
            a += 1
            w_a += f_s(2 * s - n_fingerprints)
        elif n_fingerprints - 2 * s > c_threshold:
            d += 1
            w_d += f_s(abs(2 * s - n_fingerprints))
        else:
            total_dis += 1
            total_w_dis += f_d(abs(2 * s - n_fingerprints))
    total_sim = a + d
    total_w_sim = w_a + w_d
    p = total_sim + total_dis
    w_p = total_w_sim + total_w_dis

    counters = {"a": a, "w_a": w_a, "d": d, "w_d": w_d,
                "total_sim": total_sim, "total_w_sim": total_w_sim,
                "total_dis": total_dis, "total_w_dis": total_w_dis,
                "p": p, "w_p": w_p}

    return counters


def smiles2morgan(s, radius=2, nBits=1024):
    try:
        mol = Chem.MolFromSmiles(s)
        features_vec = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=nBits)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    except:
        print('rdkit not found this smiles for morgan: ' +
              s + ' convert to all 0 features')
        features = np.zeros((nBits, ))
    return features


def smiles2MACCS(s):
    try:
        mol = Chem.MolFromSmiles(s)
        features_vec = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    except:
        print('rdkit not found this smiles for morgan: ' +
              s + ' convert to all 0 features')
        features = np.zeros((800, ))
    return features

def calculate_fingerprints(df):
    unique = pd.Series(df['SMILES'].unique()).apply(smiles2morgan)
    unique_dict = dict(zip(df['SMILES'].unique(), unique))
    df['Morgan'] = [unique_dict[i] for i in df['SMILES']]
    unique = pd.Series(df['SMILES'].unique()).apply(smiles2MACCS)
    unique_dict = dict(zip(df['SMILES'].unique(), unique))
    df['MACCS'] = [unique_dict[i] for i in df['SMILES']]
    unique = pd.Series(df['SMILES'].unique()).apply(Chem.MolFromSmiles)
    unique_dict = dict(zip(df['SMILES'].unique(), unique))
    df['ROMol'] = [unique_dict[i] for i in df['SMILES']]

    return df

# Read filename data    
df = pd.read_csv(filename, sep=" ") 
df_ref = pd.read_csv(md_db, sep=" ")

# Calculate fingerprints
df = calculate_fingerprints(df)
df_ref = calculate_fingerprints(df_ref)

# Remove NaNs
index = df['ROMol'].isna()
df = df[index==False]

# Calculate similarity
bulk_ECFP4_fps = [AllChem.GetMorganFingerprintAsBitVect(x,2) for x in df['ROMol'].dropna()]
bulk_MACCS_fps = [AllChem.GetMACCSKeysFingerprint(x) for x in df['ROMol'].dropna()]

for i in range(0,3):
    print(i)
    ref_mol = df_ref['ROMol'][i]
    
    ref_ECFP4_fps = AllChem.GetMorganFingerprintAsBitVect(ref_mol,2)
    ref_MACCS_fps = AllChem.GetMACCSKeysFingerprint(ref_mol)
    
    similarity_efcp4 = [DataStructs.FingerprintSimilarity(ref_ECFP4_fps,x) for x in bulk_ECFP4_fps]
    similatity_maccs = [DataStructs.FingerprintSimilarity(ref_MACCS_fps,x) for x in bulk_MACCS_fps]
    
    label_morgan = 'Sim_Morgan' + str(i)
    label_maccs = 'Sim_MACCS' + str(i)
    
    df[label_morgan] = similarity_efcp4
    df[label_maccs] = similatity_maccs

# Create a function to plot the molecules
def plot_molecules(colname):
    
    # Sort the dataframe by the column name
    df.sort_values(colname, ascending=False,inplace=True)
    img = PandasTools.FrameToGridImage(df.head(4), molsPerRow=2, legendsCol=colname, subImgSize=(400,400))
    plt.imshow(img)
    plt.title(colname)
    plt.show()

#Create a list of the columns to plot
list = ['Sim_Morgan0', 'Sim_MACCS0']


count = 0
# Crea for per visualizzare le immagini
for i in list:
    if count > 0:
        count = 0
    # Crea figura composta da 2 colonne con alta risoluzione
    fig, axs = plt.subplots(1, 2, figsize=(8, 8), dpi=200)
    # Eliminare gli assi
    axs[0].axis('off')
    axs[1].axis('off')
    
    #Elimina gli spazi tra i plot
    fig.subplots_adjust(wspace=0, hspace=0)
    
    # Plot nel primo sottografo immagine della molecola di riferimento ad alta risoluzione
    axs[0].imshow(PandasTools.FrameToGridImage(df_ref[count:count+1], molsPerRow=1, legendsCol='ID', subImgSize=(800,800)))
    
    plot_molecules(i)
    count = count + 1

# Save the dataframe
df_to_save = df[['SMILES','ID','Morgan','MACCS']]
df_to_save.to_pickle("Fingerprints.pkl")


# Example with more similarity indices
fingerprints = np.array([[1, 0, 1, 0],
                         [0, 0, 1, 1],
                         [1, 1, 1, 0],
                         [1, 0, 0, 0]])

# Number of fingerprints
n_fingerprints = len(fingerprints)

# Column sums
condensed = np.sum(fingerprints, axis=0)

# Generate datasets
data_sets = np.array([np.append(condensed, n_fingerprints)])

# Calculate counters
counters = calculate_counters(data_sets)

# Indices
# AC: Austin-Colwell, BUB: Baroni-Urbani-Buser, CTn: Consoni-Todschini n
# Fai: Faith, Gle: Gleason, Ja: Jaccard, Ja0: Jaccard 0-variant
# JT: Jaccard-Tanimoto, RT: Rogers-Tanimoto, RR: Russel-Rao
# SM: Sokal-Michener, SSn: Sokal-Sneath n

# Weighted Indices
ac_w = (2/np.pi) * np.arcsin(np.sqrt(counters['total_w_sim'] /
                                     counters['w_p']))
bub_w = ((counters['w_a'] * counters['w_d'])**0.5 + counters['w_a']) /\
    ((counters['w_a'] * counters['w_d'])**0.5 +
     counters['w_a'] + counters['total_w_dis'])
ct1_w = (log(1 + counters['w_a'] + counters['w_d'])) /\
    (log(1 + counters['w_p']))
ct2_w = (log(1 + counters['w_p']) - log(1 + counters['total_w_dis'])) /\
    (log(1 + counters['w_p']))
ct3_w = (log(1 + counters['w_a'])) /\
    (log(1 + counters['w_p']))
ct4_w = (log(1 + counters['w_a'])) /\
    (log(1 + counters['w_a'] + counters['total_w_dis']))
fai_w = (counters['w_a'] + 0.5 * counters['w_d']) /\
    (counters['w_p'])
gle_w = (2 * counters['w_a']) /\
    (2 * counters['w_a'] + counters['total_w_dis'])
ja_w = (3 * counters['w_a']) /\
    (3 * counters['w_a'] + counters['total_w_dis'])
ja0_w = (3 * counters['total_w_sim']) /\
    (3 * counters['total_w_sim'] + counters['total_w_dis'])
jt_w = (counters['w_a']) /\
    (counters['w_a'] + counters['total_w_dis'])
rt_w = (counters['total_w_sim']) /\
    (counters['w_p'] + counters['total_w_dis'])
rr_w = (counters['w_a']) /\
    (counters['w_p'])
sm_w = (counters['total_w_sim']) /\
    (counters['w_p'])
ss1_w = (counters['w_a']) /\
    (counters['w_a'] + 2 * counters['total_w_dis'])
ss2_w = (2 * counters['total_w_sim']) /\
    (counters['w_p'] + counters['total_w_sim'])

# Non-Weighted Indices
ac_nw = (2/np.pi) * np.arcsin(np.sqrt(counters['total_w_sim'] /
                                      counters['p']))
bub_nw = ((counters['w_a'] * counters['w_d'])**0.5 + counters['w_a']) /\
    ((counters['a'] * counters['d'])**0.5 +
     counters['a'] + counters['total_dis'])
ct1_nw = (log(1 + counters['w_a'] + counters['w_d'])) /\
    (log(1 + counters['p']))
ct2_nw = (log(1 + counters['w_p']) - log(1 + counters['total_w_dis'])) /\
    (log(1 + counters['p']))
ct3_nw = (log(1 + counters['w_a'])) /\
    (log(1 + counters['p']))
ct4_nw = (log(1 + counters['w_a'])) /\
    (log(1 + counters['a'] + counters['total_dis']))
fai_nw = (counters['w_a'] + 0.5 * counters['w_d']) /\
    (counters['p'])
gle_nw = (2 * counters['w_a']) /\
    (2 * counters['a'] + counters['total_dis'])
ja_nw = (3 * counters['w_a']) /\
    (3 * counters['a'] + counters['total_dis'])
ja0_nw = (3 * counters['total_w_sim']) /\
    (3 * counters['total_sim'] + counters['total_dis'])
jt_nw = (counters['w_a']) /\
    (counters['a'] + counters['total_dis'])
rt_nw = (counters['total_w_sim']) /\
    (counters['p'] + counters['total_dis'])
rr_nw = (counters['w_a']) /\
    (counters['p'])
sm_nw = (counters['total_w_sim']) /\
    (counters['p'])
ss1_nw = (counters['w_a']) /\
    (counters['a'] + 2 * counters['total_dis'])
ss2_nw = (2 * counters['total_w_sim']) /\
    (counters['p'] + counters['total_sim'])

# Dictionary with all the results
Indices = {'nw': {'AC': ac_nw, 'BUB': bub_nw, 'CT1': ct1_nw, 'CT2': ct2_nw, 'CT3': ct3_nw,
                  'CT4': ct4_nw, 'Fai': fai_nw, 'Gle': gle_nw, 'Ja': ja_nw,
                  'Ja0': ja0_nw, 'JT': jt_nw, 'RT': rt_nw, 'RR': rr_nw,
                  'SM': sm_nw, 'SS1': ss1_nw, 'SS2': ss2_nw},
           'w': {'AC': ac_w, 'BUB': bub_w, 'CT1': ct1_w, 'CT2': ct2_w, 'CT3': ct3_w,
                 'CT4': ct4_w, 'Fai': fai_w, 'Gle': gle_w, 'Ja': ja_w,
                 'Ja0': ja0_w, 'JT': jt_w, 'RT': rt_w, 'RR': rr_w,
                 'SM': sm_w, 'SS1': ss1_w, 'SS2': ss2_w}}

exit()
