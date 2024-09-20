# -*- coding: utf-8 -*-
"""
@author: Alexander Arteaga

"""

# An example code for classifying EEG data using EMD-SBL
from SBLEST_Block import SBLEST2, Enhanced_cov
import torch
from torch import DoubleTensor
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy.signal as signal
from scipy.stats import pearsonr
# Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tau =0
K = 1
Epoch = 5000
num_folds = 10
kf = KFold(n_splits=num_folds, shuffle=True)
#%%
#Reverse Time data augmentation
def reverse_time(eeg_data,paug):
    step = 125
    window_size=125
    num_subjects, num_channels, num_time_points = eeg_data.shape
    num_segments = list(range(0,num_time_points,step))
    for i in range(num_subjects):
        for j in num_segments:
            if np.random.rand() < paug:
                if j +window_size<= num_time_points:  #make sure we don't go past time
                    time_window = eeg_data[i,:,j:j+window_size]
                    
                    reversed_window = time_window[:,::-1]
                    eeg_data[i,:,j:j+window_size] = reversed_window
                
    return eeg_data
#%%
#bandpass filter
def filter_signal(input_signal, low_freq, high_freq, sampling_rate):
    nyquist = 0.5 * sampling_rate
    low = low_freq/nyquist
    high = high_freq/nyquist
    b,a=signal.butter(2, [low,high], btype='band')
    filtered_signal=signal.filtfilt(b,a,input_signal)
    return filtered_signal
#%%
#assuming decomposed signals using EMD
def prepare_data(X_train,num_channels):
    X= X_train[:,0:num_channels,:]
    X1 = X_train[:,num_channels:num_channels*2,:]
    X2 = X_train[:,num_channels*2:num_channels*3,:]

    #Transpose to Channels, Timepoints, Subjects
    X = np.transpose(X, (1, 2, 0))
    X1 = np.transpose(X1, (1, 2, 0))
    X2 = np.transpose(X2, (1, 2, 0))
    
    X = DoubleTensor(X).to(device)
    X1 = DoubleTensor(X1).to(device)
    X2 = DoubleTensor(X2).to(device)

    return(X,X1,X2)
#%%
#Assuming just the original time series. Here we use a bandpass filter to create additional oscillatory 
#components. 
def prepare_data_fft (X):
    
    #bandpass to beta
    X1= filter_signal(X, 13, 30, 250)
    #bandpass to alpha
    X2 = filter_signal(X, 8, 13,250)
    #Transpose to C, T, M
    X = np.transpose(X, (1, 2, 0))
    X1 = np.transpose(X1, (1, 2, 0))
    X2 = np.transpose(X2, (1, 2, 0))
    
    X = DoubleTensor(X).to(device)
    X1 = DoubleTensor(X1).to(device)
    X2 = DoubleTensor(X2).to(device)

    return(X,X1,X2)
#%%
def visualize(y_test_store,y_pred_store):
    #Simple performance visualization
    y_test_store = y_test_store.reshape(-1)
    r, p =pearsonr(y_test_store, y_pred_store)
    r_2 = r2_score(y_test_store, y_pred_store)
    plt.scatter(y_test_store, y_pred_store, marker='.',color='darkcyan')
    plt.axis('square')
    y_test_store = y_test_store.reshape(-1,1)
    y_pred_store = y_pred_store.reshape(-1,1)
    perf_model=LinearRegression().fit(y_test_store, y_pred_store)
    plt.plot(y_test_store, perf_model.predict(y_test_store), linewidth=2)
    #labels to the figure
    plt.xlabel('Actual bdi progression')
    plt.ylabel('Predicted bdi progression')
    plt.text(2, 29, f'R2 = {r_2}\nR ={r}\np={p}', fontsize=12,bbox=dict(facecolor='white', alpha=0.5))
    plt.ylim(-10,45)
    plt.xlim(-10,45)
    plt.title('Treatment Outcome Prediction')
    plt.show()

#%%
#Load data with its decomposed components. Here I am assuming the data is a numpy array size
# M,C * 3,T. Where M is number of subjects, C is number of channels( multiplied by three because we include
# two decomposed oscillation components plus the original time series: Original, IMF1, IMF2), T is timepoints.
  
data_loc = "Insert/path/to/data"
label_loc ="Insert/path/to/target/label"
data = np.load(data_loc)
labels = np.load(label_loc)

#%%
if __name__ == '__main__':

    y_test_store=[]
    y_pred_store=[]
    fold=0
    #10 fold cv
    for train_index, test_index in kf.split(data):
        print(f"Processing subject {fold + 1}...")
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        #Time reverse data augmentation
        X_train = np.vstack([X_train, reverse_time(X_train,0.2),reverse_time(X_train,0.2)])
        y_train = np.vstack([y_train, y_train,y_train])

        #Data preparation for input into model.
        X,X1,X2 = prepare_data(X_train,num_channels=26)
        X_test_orig,X_test1,X_test2 =prepare_data(X_test,num_channels=26)
        
        # Here we calculate the residual oscillation. In my case it is original signal - IMF1 - IMF2
        X = X-X1-X2
        X_test_orig = X_test_orig - X_test1 - X_test2
        
        #training labels to device
        Y_train = DoubleTensor(y_train).to(device)
        
    
        # Training stage: run SBLEST on the training set
        print('\n', 'FIR filter order: ', str(K), '      Time delay: ', str(tau))
        
        W, alpha, V, Wh= SBLEST2(X,X1,X2, Y_train, K, tau, Epoch)


        R_test,_ = Enhanced_cov(X_test_orig,X_test1,X_test2, K, tau, Wh, train=0)

        # vec operation to calculate predictions. 
        vec_W = W.T.flatten()   
        predict_Y = R_test @ vec_W
        #Store both predicted and test labels
        y_pred = predict_Y.cpu().numpy()  
        y_pred_store.append(y_pred)
        y_test_store.append(y_test)

        fold +=1
    y_pred_store =np.concatenate(y_pred_store)
    y_test_store =np.concatenate(y_test_store)
    #visualize performance of 1 run
    visualize(y_test_store, y_pred_store)