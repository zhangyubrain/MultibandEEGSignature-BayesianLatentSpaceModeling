# -*- coding: utf-8 -*-
"""
@author: Alexander Arteaga

"""

# An example code for classifying single-trial EEG data using SBLEST
import numpy as np
import emd

#%%
def decompose(X,sample_rate):
    X_imf1 = np.zeros_like(X)
    X_imf2= np.zeros_like(X)
    #X_imf3 = np.zeros_like(X)
    #X_imf4 = np.zeros_like(X)


    for x in range(X.shape[0]):
        print(f'decomposing EEG of subject{x}')
        for c in range(X.shape[1]):
            imf = emd.sift.iterated_mask_sift(X[x][c], sample_rate=sample_rate, max_imfs=4)
            imf_new = imf.transpose()
            X_imf1[x][c] = imf_new[0]
            X_imf2[x][c] = imf_new[1]
            #X_imf3[x][c] = imf_new[2]
            #X_imf4[x][c] = imf_new[3]

    X_final= np.concatenate((X,X_imf1,X_imf2), axis=1)        
    return(X_final)
#%%


if __name__ == '__main__':
    
    # Data should be formatted into a 3D array: Subjects, Channels, Timepoints
    #In this example, it is assumed that the data can be decomposed into four distinct bands and only 
    #the first two components (bands, IMF1, IMF2) are stored. 
    data=np.load('Insert/EEG/data/location')
    print('decomposing')
    data = decompose(data,sample_rate = 250)
    print('done')
    

    np.save('Insert/HD/location', data)



