
# Let us import all the necessary libraries

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from pyhht import EMD
from scipy.stats import norm
from scipy.stats import entropy
from tqdm import tqdm

print("All imports done")

def read_data(filename):
    x = pickle._Unpickler(open(filename, 'rb'))
    x.encoding = 'latin1'
    data = x.load()
    return data

labels = []
data = []
for i in tqdm(range(32)): 
    fileph =  "../deap_data/s" + format(i+1, '02') + ".dat"
    d = read_data(fileph)
    labels.append(d['labels'])
    data.append(d['data'])  #32,40,40,8064
    
data=np.array(data)


# Data for the 1st participant and 1st trial

eeg_data=data[0, :5, :32, 384:8064]

"""
EMD object is an instance of the EMD class that represents the EMD algorithm. 
The EMD object can be used to decompose a signal into its IMFs using the decompose() method. 
The resulting IMFs can be accessed using the imfs attribute of the EMD object, 
which returns a 2D array where each row represents an IMF.
"""

decomposer = EMD(eeg_data.ravel())
imfs = decomposer.decompose()

# Resizing the IMFs to their original shape of the signals (eeg_data)

imf1=imfs[0].reshape(32,7680)
imf2=imfs[1].reshape(32,7680)
imf3=imfs[2].reshape(32,7680)
imf4=imfs[3].reshape(32,7680)
imf5=imfs[4].reshape(32,7680)

# Applying sliding window to the imf data for 5 IMFs

eeg_imf1=[[]*29]*len(imf1)
eeg_imf2=[[]*29]*len(imf2)
eeg_imf3=[[]*29]*len(imf3)
eeg_imf4=[[]*29]*len(imf4)
eeg_imf5=[[]*29]*len(imf5)
for i in range(1,len(imf1)+1):
    
    if (2*(i-1)+4)<=60:
        eeg_imf1[i-1].append(imf1[i-1][2*(i-1)*128:(2*(i-1)+4)*128])
        eeg_imf2[i-1].append(imf2[i-1][2*(i-1)*128:(2*(i-1)+4)*128])
        eeg_imf3[i-1].append(imf3[i-1][2*(i-1)*128:(2*(i-1)+4)*128])
        eeg_imf4[i-1].append(imf4[i-1][2*(i-1)*128:(2*(i-1)+4)*128])
        eeg_imf5[i-1].append(imf5[i-1][2*(i-1)*128:(2*(i-1)+4)*128])
        

eeg_imf1=np.array(eeg_imf1) #(32, 29, 512)
eeg_imf2=np.array(eeg_imf2)
eeg_imf3=np.array(eeg_imf3)
eeg_imf4=np.array(eeg_imf4)
eeg_imf5=np.array(eeg_imf5)

# Declaring Gaussian Normalization function and applying to the eeg_imf data

def gaussian_normalization(data):
    mean = np.mean(data)
    std = np.std(data)
    gaussian = norm.cdf(data, mean, std)
    return gaussian

eeg_imf1=gaussian_normalization(eeg_imf1)
eeg_imf2=gaussian_normalization(eeg_imf2)
eeg_imf3=gaussian_normalization(eeg_imf3)
eeg_imf4=gaussian_normalization(eeg_imf4)
eeg_imf5=gaussian_normalization(eeg_imf5)

# Extracting the entropies features from the eeg_imf data

entropies1 = []   #29*32
entropies2 = []
entropies3 = []
entropies4 = []
entropies5 = []

for i in range(1,eeg_imf1.shape[0]+1):
    for j in range(1,eeg_imf1.shape[1]+1):
        prob_dist = np.histogram(eeg_imf1[i-1,j-1, :], bins='fd')[1] / eeg_imf1.shape[2]
        entropies1.append(entropy(prob_dist, base=2))
        prob_dist = np.histogram(eeg_imf2[i-1,j-1, :], bins='fd')[1] / eeg_imf2.shape[2]
        entropies2.append(entropy(prob_dist, base=2))
        prob_dist = np.histogram(eeg_imf3[i-1,j-1, :], bins='fd')[1] / eeg_imf3.shape[2]
        entropies3.append(entropy(prob_dist, base=2))
        prob_dist = np.histogram(eeg_imf4[i-1,j-1, :], bins='fd')[1] / eeg_imf4.shape[2]
        entropies4.append(entropy(prob_dist, base=2))
        prob_dist = np.histogram(eeg_imf5[i-1,j-1, :], bins='fd')[1] / eeg_imf5.shape[2]
        entropies5.append(entropy(prob_dist, base=2))
    
    
# Concatenating the entropies and reshaping

resen=entropies1+entropies2+entropies3+entropies4+entropies5
resen=np.array(resen).reshape(1,29,160)
np.save('entropies_1.npy', resen)

print(resen.shape)
