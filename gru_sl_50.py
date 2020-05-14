#%%
import numpy as np
import sys
import time
import os

#Functions
# 1. Models
from tgan import tgan

# 2. Data Loading
from data_loading import google_data_loading, sine_data_generation, soro_data_loading, reverse_soro_data_loading

# 3. Metrics
sys.path.append('metrics')
from discriminative_score_metrics import discriminative_score_metrics
from visualization_metrics import PCA_Analysis, tSNE_Analysis
from predictive_score_metrics import predictive_score_metrics


print('Finish importing necessary packages and functions')

# %%
# Main Parameters
start_time = time.time()
# Data
data_set = ['google','sine', 'soro']
data_name = data_set[2]

# Experiments iterations
Iteration = 1 # 2
Sub_Iteration = 3 # 3

#Data Loading
seq_length = 50 # 50

if data_name == 'google':
    dataX = google_data_loading(seq_length)
elif data_name == 'sine':
    No = 10000
    F_No = 5
    dataX = sine_data_generation(No, seq_length, F_No)
elif data_name == 'soro':
    dataX, scaler = soro_data_loading(seq_length)

print(data_name + ' dataset is ready.')
print(np.shape(dataX))

# %%
# Newtork Parameters
parameters = dict()

parameters['hidden_dim'] = len(dataX[0][0,:]) * 4
parameters['num_layers'] = 3
parameters['iterations'] = 10000 # 50000
parameters['batch_size'] = 128
parameters['module_name'] = 'gru'   # Other options: 'lstm' or 'lstmLN'
parameters['z_dim'] = len(dataX[0][0,:]) 

print('Parameters are ' + str(parameters))

#Experiments
# Output Initialization
Discriminative_Score = list()
Predictive_Score = list()

#%%
print('Start iterations') 
    
# Each Iteration
for it in range(Iteration):

    print("Starting Iteration {}/{}".format(it, Iteration))
    # Synthetic Data Generation
    dataX_hat = tgan(dataX, parameters)   
    print('Finish Synthetic Data Generation') 

    # Calculate Performance Metrics
    # 1. Discriminative Score
    Acc = list()
    for tt in range(Sub_Iteration):
        Temp_Disc = discriminative_score_metrics (dataX, dataX_hat)
        Acc.append(Temp_Disc)

    Discriminative_Score.append(np.mean(Acc))

    # 2. Predictive Performance
    MAE_All = list()
    for tt in range(Sub_Iteration):
        MAE_All.append(predictive_score_metrics (dataX, dataX_hat))
        
    Predictive_Score.append(np.mean(MAE_All))     
    print("Finished Computing Performance Metrics")

print('Finish TGAN iterations')

#%%
# tSNE AND PCA
tSNE_Analysis (dataX, dataX_hat, os.path.join("hyperparameter_tests/gru_sl_50", data_name+"_tSNE.png"))
PCA_Analysis (dataX, dataX_hat, os.path.join("hyperparameter_tests/gru_sl_50", data_name+"_PCA.png"))
print("Finished tSNE and PCA analysis")

#%%
# invert scaling
dataX_hat = reverse_soro_data_loading(dataX_hat, seq_length)
dataX_hat = scaler.inverse_transform(dataX_hat)

dataX = reverse_soro_data_loading(dataX, seq_length)
dataX = scaler.inverse_transform(dataX)
print("Finished Data Inversion")

#%%
# Save outputs
np.save(os.path.join("hyperparameter_tests/gru_sl_50", "soro_synthetic_data"), dataX_hat)

with open(os.path.join("hyperparameter_tests/gru_sl_50", "synthetic_dpScore.txt"), "w+") as f:
    f.write('Discriminative Score - Mean: ' + str(np.round(np.mean(Discriminative_Score),4)) + ', Std: ' + str(np.round(np.std(Discriminative_Score),4)))
    f.write('\nPredictive Score - Mean: ' + str(np.round(np.mean(Predictive_Score),4)) + ', Std: ' + str(np.round(np.std(Predictive_Score),4)))

print("Finished Saving Data")
print(time.time() - start_time, "seconds")
#%%
# Visualization
from matplotlib import pyplot

data = [dataX, dataX_hat]
titles = ["dataX", "dataX_hat"]
# plot each column
pyplot.figure()
for i in range(len(data)):
  pyplot.subplot(3, 1, i+1)
  pyplot.plot(data[i])
  pyplot.title(titles[i], y=0.5, loc='right')
  pyplot.savefig(os.path.join("hyperparameter_tests/gru_sl_50", titles))
# %%
