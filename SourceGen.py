#
# Generate sliprates and fault coordinates files for DG
#
# John Diaz July 2022

# To Do List:
#    Filter sliprates 
#    Sliprate animation
#

import os
import warnings

import numpy as np
import pickle
import matplotlib.pyplot as plt
#from binaryfile import BinaryFile
from scipy import signal
import matplotlib.cm as cm
from scipy.io import FortranFile

from IPython import get_ipython
get_ipython().magic('reset -sf')
warnings.filterwarnings("ignore")
plt.close('all') 
os.system('clear')

# Source Inputs
tmax = 10.0;
dt = 0.01;
nt = int(tmax/dt)+1
lowcut  = 2.0                              # srate low pass cut frecuency
highcut = 0.01                             # srate cut high pass frecuency
fs = 1/dt                                  # Sample rate
rake = 90

# ID of simulation
ID = 1

# Name of the Fault file
ObjName = 'Outputs/3DFaults/SIVBenchmarkDsef0_dhF500m.pickle'

print("  ")
print(" START PROGRAM ")
print("  ")

# Load the Fault object
with open(ObjName, 'rb') as handle:
    Fault = pickle.load(handle)
 
nstk = Fault['nstk']
ndip = Fault['ndip']

nflt = nstk*ndip
SlipMat  = np.reshape(Fault['SlipMat'], (ndip, nstk))
Slipmean = np.mean(SlipMat)
RupTMat  = np.reshape(Fault['RupTMat'], (ndip, nstk))
RiseTMat = np.reshape(Fault['RiseTMat'], (ndip, nstk)) 

stkMat = np.reshape(Fault['stkMat'], (ndip, nstk))
dipMat = np.reshape(Fault['dipMat'], (ndip, nstk))

SRate_nonfilt = np.zeros((ndip,nstk,nt))
SRate = np.zeros((ndip,nstk,nt))
maxsr = np.zeros((ndip,nstk))

it_start = np.round(RupTMat/dt)
it_start = it_start.astype(int)
it_end   = np.round((RupTMat+RiseTMat)/dt)
it_end   = it_end.astype(int)

# Coefs fol filtering
w = lowcut/(fs/2)                         # Normalize the frequency
b, a = signal.butter(4, w, 'low')

for idip in range (0,ndip):
    for istk in range (0,nstk):
        iti = it_start[idip,istk]
        itf = it_end[idip,istk]
        SRate_nonfilt[idip,istk,iti:itf] = SlipMat[idip,istk]/RiseTMat[idip,istk]
        # Filtering with lowpass the sliprate
        SRate[idip,istk,:] =  SRate_nonfilt[idip,istk,:]
        #SRate[idip,istk,:] =  signal.filtfilt(b, a, SRate_nonfilt[idip,istk,:])
        maxsr[idip,istk] = np.amax(SRate[idip,istk,:])

# Slip positive in the direction of dip and in the direction of strike
SRdip = (-SRate.flatten(order='F'))*np.sin(np.deg2rad(rake))
SRstk = (SRate.flatten(order='F'))*np.cos(np.deg2rad(rake))
SRdip = np.float32(SRdip)
SRstk = np.float32(SRstk)
Nsr = np.arange(0,SRdip.size,1)

outname = Fault['outname']

SRdipName = 'Outputs/3DFaults/srate_dip_'+outname+'_ID_'+str(ID)
SRstkName = 'Outputs/3DFaults/srate_str_'+outname+'_ID_'+str(ID)

# Write srate files in binary files for fortran
# fsrd = BinaryFile(SRdipName, mode="w", order="fortran")
# fsrd.write(SRdip)
# fsrd.close()
# fstk = BinaryFile(SRstkName, mode="w", order="fortran")
# fstk.write(SRstk)
# fstk.close()

fsrd = FortranFile(SRdipName, 'w')
fsrd.write_record(SRdip)
fsrs = FortranFile(SRstkName, 'w')
fsrs.write_record(SRstk)

# Write fccor file
fcoorHeader = "%d  %d %4.2f " %(nflt, nt, dt) 
fcoor = np.array(Fault['fcoor'])
fcoorName = 'Outputs/3DFaults/fcoor_'+outname+'_ID_'+str(ID)+'.in'

with open(fcoorName,'wb') as f:
    np.savetxt(f, fcoor, header=fcoorHeader, comments=' ',fmt = '%9.4f')  
    
print(f" Coordinates saved in file: {fcoorName}" )
print(f" SlipRate dip saved in file: {SRdipName}" )
print(f" SlipRate stk saved in file: {SRstkName}" )

fig = plt.figure()
for it in range(0,nt,50):
    SRNF = SRate_nonfilt[:,:,it]
    SRF  = SRate[:,:,it]
    ax1 = fig.add_subplot(121)
    ax1.pcolormesh(stkMat,dipMat,SRNF,cmap=cm.seismic)
    ax2 = fig.add_subplot(122)
    ax2.pcolormesh(stkMat,dipMat,SRF,cmap=cm.seismic)
    plt.pause(0.01)
    #ax[0,1].pcolormesh(Fault.stkMat,Fault.dipMat,SRF,cmap=cm.seismic)
    #plt.colorbar(Vm,label='Vx (m/s)',orientation="horizontal",shrink=.68)
    # ax.set_aspect('equal',adjustable='box')
    # ax.set_xlabel(' Strike (Km) ')
    # ax.set_ylabel(' Dip (Km) ')



fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(SRdip)
ax1 = fig.add_subplot(122)
ax1.plot(SRstk)

print("  ")
print(" END PROGRAM ")
print("  ")
