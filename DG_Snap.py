#
# Generate Snapshoots of the surface propagation
#
# John Diaz August 2022

# To Do List:
#    Filtering 
#   Same colorbar
   
import os
import warnings

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from pathlib import Path

from scipy import signal
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

from IPython import get_ipython
get_ipython().magic('reset -sf')
warnings.filterwarnings("ignore")
plt.close('all') 
os.system('clear')

# DG folder
DGFolder = Path('DGrun/SIVBenchmarkDsef0_dhF500m_ID1_5.0Hz/')

# Space coordinates (km)
# SIV Benchmark
xini = -40.0
xend = 40.0
yini = -40.0
yend = 40.0

# Maximum Velocity for colorbar (m/s)
Vmax = 5.0

# Simulation Time (s), Spatial step (Km) and Time (s) step
time = 10.0
dx = 0.1
dt = 0.1

# Filtering parameters
lowcut  = 1.0                              # srate low pass cut frecuency
highcut = 0.01                             # srate cut high pass frecuency
fs = 1/dt                                  # Sample rate

print("  ")
print(" START PROGRAM ")
print("  ")

# Number of snaps shots
#nsnap = int(time/dt) 

xend = xend+dx
yend = yend+dx
X = np.arange(xini,xend,dx)
Y = np.arange(yini,yend,dx)

nx = X.size
ny = Y.size

xMat, yMat = np.meshgrid(X,Y)
zMat = np.zeros((nx,ny))

# # Read binary files from fortran
# fvs = DGFolder.joinpath('vs.map')
# print(f" Loading vs file: {fvs} ")
# vs = BinaryFile(fvs, mode="r", order="fortran")
# SVelo=vs.read(dtype='float32',shape=(nx,ny))

# vscolor=cm.ScalarMappable(cmap=cm.jet)
# vscolor.set_array(np.linspace(np.min(np.unique(SVelo)),max(np.unique(SVelo)),100))
# VSColors = Myplots.SetColorbar(SVelo)

# fig = plt.figure()
# plt.pcolormesh(xMat,yMat,SVelo,facecolors=VSColors)
# fig.colorbar(vscolor,label='Vs (m/s)', orientation="horizontal",shrink=.95)

# Read binary files from fortran
fnameDGx = DGFolder.joinpath('VX.snap')
fnameDGy = DGFolder.joinpath('VY.snap')

print(" Loading snap files:\n ")
print(f" {fnameDGx}")
print(f" {fnameDGy}")

Vx = np.fromfile(fnameDGx, dtype=np.float32)
Vy = np.fromfile(fnameDGy, dtype=np.float32)

nsnap = int(Vx.size/(nx*ny))

Vx = np.reshape(Vx,(nx,ny,nsnap),order = 'F')
Vy = np.reshape(Vy,(nx,ny,nsnap),order = 'F')

Vcolor = np.linspace(-Vmax,Vmax,1000)
Vm = cm.ScalarMappable(cmap=cm.seismic)
Vm.set_array(Vcolor)

# Filtering Snapshots
# Coefs fol filtering
w = lowcut/(fs/2)                         # Normalize the frequency
b, a = signal.butter(4, w, 'low')
VxF = np.zeros((nx,ny,nsnap))
VyF = np.zeros((nx,ny,nsnap))

for ix in range (0,nx):
    for iy in range (0,ny):
        VxF[ix,iy,:] = signal.filtfilt(b, a, Vx[ix,iy,:])
        VyF[ix,iy,:] = signal.filtfilt(b, a, Vy[ix,iy,:])
        
        
for isnap in range(0,nsnap):
    fig = plt.figure(figsize=(12,12))
    ax = fig.subplots(1,1)
    V = VxF[:,:,isnap]
    ax.pcolormesh(xMat,yMat,V,cmap=cm.seismic)
    plt.colorbar(Vm,label='Vx (m/s)',orientation="horizontal",shrink=.68)
    ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel(' X (Km) ')
    ax.set_ylabel(' Y (Km) ')

# #fig, ax = plt.figure(figsize=(12,12))
# fig, ax = plt.subplots()
# mesh = ax.pcolormesh(Vx[:, :, 15], vmin=-Vmax, vmax=Vmax)

# def animate_vx(i):
#     mesh.set_array(Vx[:, :, i].ravel())
#     return mesh

# anim = FuncAnimation(fig, animate_vx, interval=10, frames=nsnap, repeat=False)

# # saving to m4 using ffmpeg writer
# writervideo = animation.FFMpegWriter(fps=60)
# anim.save('SIVBenchmark_dhF500m.mp4', writer=writervideo)
# plt.close()


# for isnap in range(0,nsnap):
#     fig = plt.figure(figsize=(12,12))
#     ax = fig.subplots(1,1)
#     V = Vx[:,:,isnap]
#     ax.pcolormesh(xMat,yMat,V,cmap=cm.seismic)
#     plt.colorbar(Vm,label='Vx (m/s)',orientation="horizontal",shrink=.68)
#     ax.set_aspect('equal',adjustable='box')
#     ax.set_xlabel(' X (Km) ')
#     ax.set_ylabel(' Y (Km) ')
    
# fig = plt.figure(figsize=(12,12))  
# ax = fig.subplots(1,1)
# def animation_vx(i):
#     V = Vx[:,:,i]
#     ax.pcolormesh(xMat,yMat,V,cmap=cm.seismic)
#     plt.colorbar(Vm,label='Vx (m/s)',orientation="horizontal",shrink=.68)
#     ax.set_aspect('equal',adjustable='box')
#     ax.set_xlabel(' X (Km) ')
#     ax.set_ylabel(' Y (Km) ')
#     return ax
 

   
#animation = FuncAnimation(fig, animation_vx,interval = nsnap)
# for isnap in range(0,nsnap):
#     fig = plt.figure()
#     ax = fig.subplots(1,1)
#     V = Vy[:,:,isnap]
#     ax.pcolormesh(xMat,yMat,V,cmap=cm.seismic)
#     plt.colorbar(Vm,label='Vy (m/s)',orientation="horizontal",shrink=.9)