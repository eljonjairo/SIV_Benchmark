#
# Compare velocity fro SIVBenchmark
#
# John Diaz August 2022

# To Do List:
# Add Data
    
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy import signal
from scipy.io import FortranFile


from IPython import get_ipython
get_ipython().magic('reset -sf')
warnings.filterwarnings("ignore")
plt.close('all') 
os.system('clear')

# DG folder
DGFolder = Path('DGrun/SIVBenchmarkDsef0_dhF500m_ID1_1.0Hz/')

# Data folder
DataFolder = Path('Data/')

label   = 'dsef0'
modeler = 'CausseCompsyn'

nstats = 20
nt_DG  = 201
dt_DG  = 0.0495587

dt = 0.05   # Output dt

# Filtering parameters
lowcut  = 1.0                              # low pass cut frecuency
highcut = 0.1                              # cut high pass frecuency
forder  = 4                                # Butterworth order

# initial time for plotting components (s)
tiy = 5;  tix = 30;  tiz = 55;

print("  ")
print(" START PROGRAM ")
print("  ")

fileFSx = fnameDGx = DGFolder.joinpath('VX_1')
fileFSy = fnameDGx = DGFolder.joinpath('VY_1')
fileFSz = fnameDGx = DGFolder.joinpath('VZ_1')

print(" Loading DG velocity files:\n ")
print(f" {fileFSx}")
print(f" {fileFSy}")
print(f" {fileFSz}")

Vsyn_FSx = np.reshape(np.fromfile(fileFSx, dtype=np.float32),(nstats,nt_DG), order ='F')
Vsyn_FSy = np.reshape(np.fromfile(fileFSy, dtype=np.float32),(nstats,nt_DG), order ='F')
Vsyn_FSz = np.reshape(np.fromfile(fileFSz, dtype=np.float32),(nstats,nt_DG), order ='F')

# DG Time Vector
time_DG = np.linspace(0, dt_DG*(nt_DG-1), nt_DG)

# Coefs for DG filtering
fs_DG   = 1/dt_DG
wlow_DG = lowcut/(fs_DG/2)                         # Normalize the frequency
whp_DG  = highcut/(fs_DG/2)
blow_DG, alow_DG = signal.butter(forder, wlow_DG, 'low')
bhp_DG, ahp_DG   = signal.butter(forder, whp_DG, 'hp')

print()
print(' Loading Syntetic Velocity files:')
print()

for istat in range (0,nstats):
    # Synthetic Data Proccesing
    DataFile = DataFolder.joinpath(label+'_'+modeler+'_'+str(istat+1)+'.syn')
    print(f'{DataFile}')
    with open(DataFile, 'r') as f: lines = f.readlines()[2]
    Head = lines.strip().split(' ')
    Head = [ele for ele in Head if ele.strip()]
    nt_SYN = int(Head[0])
    dt_SYN = float(Head[1])
    time_SYN = np.linspace(0, dt_SYN*(nt_SYN-1), nt_SYN)
    # Coefs for SYN filtering
    fs_SYN = 1/dt_SYN
    wlow_SYN = lowcut/(fs_SYN/2)                         # Normalize the frequency
    whp_SYN  = highcut/(fs_SYN/2)
    blow_SYN, alow_SYN = signal.butter(forder, wlow_SYN, 'low')
    bhp_SYN, ahp_SYN = signal.butter(forder, whp_SYN, 'hp')
    Data = pd.read_csv(DataFile, skiprows=4, names= ['X', 'Y', 'Z'], sep='\s+', dtype={'X': np.float64, "Y": np.float64, "Z": np.float64})
    velox_SYN = Data.loc[:,'X']
    veloy_SYN = Data.loc[:,'Y']
    veloz_SYN = Data.loc[:,'Z']
    # low pass filtering
    SYNvelox_low = signal.filtfilt(blow_SYN, alow_SYN, velox_SYN)
    SYNveloy_low = signal.filtfilt(blow_SYN, alow_SYN, veloy_SYN)
    SYNveloz_low = signal.filtfilt(blow_SYN, alow_SYN, veloz_SYN)
    # high pass filtering
    SYNvelox = signal.filtfilt(bhp_SYN, ahp_SYN, SYNvelox_low)
    SYNveloy = signal.filtfilt(bhp_SYN, ahp_SYN, SYNveloy_low)
    SYNveloz = signal.filtfilt(bhp_SYN, ahp_SYN, SYNveloz_low)

    # DG Data Proccesing
    velox_DG = Vsyn_FSx[istat,:]
    veloy_DG = Vsyn_FSy[istat,:]
    veloz_DG = Vsyn_FSz[istat,:]
    # low pass filtering
    DGvelox_low = signal.filtfilt(blow_DG, alow_DG, velox_DG)
    DGveloy_low = signal.filtfilt(blow_DG, alow_DG, veloy_DG)
    DGveloz_low = signal.filtfilt(blow_DG, alow_DG, veloz_DG)
    # high pass filtering
    DGvelox = signal.filtfilt(bhp_DG, ahp_DG, DGvelox_low)
    DGveloy = signal.filtfilt(bhp_DG, ahp_DG, DGveloy_low)
    DGveloz = signal.filtfilt(bhp_DG, ahp_DG, DGveloz_low)

    di = (istat-1)*0.3
    
    fig = plt.figure(1)
    plt.plot(time_DG+tiy,DGveloy-di,color='k')
    plt.plot(time_DG+tix,DGvelox-di,color='k')
    plt.plot(time_DG+tiz,DGveloz-di,color='k')
    plt.plot(time_SYN+tiy,SYNveloy-di,color='r')
    plt.plot(time_SYN+tix,SYNvelox-di,color='r')
    plt.plot(time_SYN+tiz,SYNveloz-di,color='r')
    
