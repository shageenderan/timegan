'''
2019 NeurIPS Submission
Title: Time-series Generative Adversarial Networks
Authors: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar

Last Updated Date: May 29th 2019
Code Author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

Data loading
(1) Load Google dataset
- Transform the raw data to preprocessed data
(2) Generate Sine dataset

Inputs
(1) Google dataset
- Raw data
- seq_length: Sequence Length
(2) Sine dataset
- No: Sample Number
- T_No: Sequence Length
- F_No: Feature Number

Outputs
- time-series preprocessed data
'''

#%% Necessary Packages
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot
# Min Max Normalizer
from sklearn.preprocessing import MinMaxScaler

def _MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

#%% Load Google Data
    
def google_data_loading (seq_length):

    # Load Google Data
    x = np.loadtxt('data/GOOGLE_BIG.csv', delimiter = ",",skiprows = 1)
    # Flip the data to make chronological data
    x = x[::-1]
    # Min-Max Normalizer
    x = _MinMaxScaler(x)
    
    # Build dataset
    dataX = []
    
    # Cut data by sequence length
    for i in range(0, len(x) - seq_length):
        _x = x[i:i + seq_length]
        dataX.append(_x)
        
    # Mix Data (to make it similar to i.i.d)
    idx = np.random.permutation(len(dataX))
    
    outputX = []
    for i in range(len(dataX)):
        outputX.append(dataX[idx[i]])
    
    return outputX
  
#%% Sine Data Generation

def sine_data_generation (No, T_No, F_No):
  
    # Initialize the output
    dataX = list()

    # Generate sine data
    for i in range(No):
      
        # Initialize each time-series
        Temp = list()

        # For each feature
        for k in range(F_No):              
                          
            # Randomly drawn frequence and phase
            freq1 = np.random.uniform(0,0.1)            
            phase1 = np.random.uniform(0,0.1)
          
            # Generate Sine Signal
            Temp1 = [np.sin(freq1 * j + phase1) for j in range(T_No)] 
            Temp.append(Temp1)
        
        # Align row/column
        Temp = np.transpose(np.asarray(Temp))
        
        # Normalize to [0,1]
        Temp = (Temp + 1)*0.5
        
        dataX.append(Temp)
                
    return dataX
    
#%% Load SORO Data 
def soro_data_loading(seq_length):
    raw_data = loadmat('data/train.mat')
    
    _x = np.dstack((raw_data["Flex"], raw_data["Force"], raw_data["Pressure"]))
    _x = _x.reshape(-1,3)

    x_markers = np.array(raw_data["X_markers"])
    y_markers = np.array(raw_data["Y_markers"])

    x = np.zeros((_x.shape[0], _x.shape[1] + x_markers.shape[1] + y_markers.shape[1]))
    for i in range(len(_x)):
        x[i] = np.append(_x[i], (x_markers[i], y_markers[i]))

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    
    dataX = []
    # Cut data by sequence length
    for i in range(0, len(x) - seq_length):
        _x = x[i:i + seq_length]
        dataX.append(_x)

    return dataX, scaler
# %% Load SORO Data differently
def soro_data_loading_diff(seq_length):
    raw_data = loadmat('data/train.mat')
    
    x = list(zip(raw_data["Flex"], raw_data["Force"], raw_data["Pressure"]))
    x = np.array(x).reshape(-1,3)
    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)

    Xs = []
    for i in range(0, len(x),seq_length):
        if i + seq_length > len(x):
            curr = np.array(x[i:(i + seq_length)])
            # to_pad = 50 - len(curr)
            print("dropping {} data".format(len(curr)))
            # padding = np.zeros((to_pad, curr.shape[1]))
            # Xs.append(np.concatenate([curr, padding]))
        else:
            v = x[i:(i + seq_length)]
            Xs.append(v)
    dataX = np.array(Xs)
    return dataX.reshape(dataX.shape[0], dataX.shape[1], -1), scaler
# %%
def reverse_soro_data_loading(data, seq_length):
    head = data[0]
    tail = np.array([f[seq_length-1] for f in data])
    return np.concatenate((head, tail))
# x, scaler , ori = soro_data_loading(50)
# x_inv = scaler.inverse_transform(reverse_soro_data_loading(x, 50))

# print(np.shape(x))
# print(np.shape(ori))
# print(np.shape(x_inv))

# print(ori[-2], x_inv[-1])

# pyplot.figure()
# pyplot.subplot(3, 1, 1)
# pyplot.plot(ori)
# pyplot.show()


# pyplot.figure()
# pyplot.subplot(3, 1, 1)
# pyplot.plot(x_inv)
# pyplot.show()



# %%
