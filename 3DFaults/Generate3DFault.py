#
# Generate Fault output files
#
# John Diaz July 2022

# To Do List:
#    Add index from hypocenter coordinates
#    Test python dealunay triangulation.
#

import os
import warnings
from IPython import get_ipython

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import interpolate

from scipy.spatial import Delaunay
import pickle
from pathlib import Path

warnings.filterwarnings("ignore")
os.system('clear')
get_ipython().magic('reset -sf')

dhF = 0.5  # Output subfaults size in Km
inDir  = Path('../Input/')
outDir = Path('../Outputs/')
infile = 'RuptureModel_DSEF0.dat'  # Input file with fault data
name = "SIVBenchmarkDsef0_dhF"  # Output files name

# Dimension of input fault
nstkin = 13
ndipin = 13

# X and Y limits (Km) for plots
xmin = -5.0
xmax = 20.0
ymin = -15.0
ymax = 15.0
zmin = -20.0
zmax = 0.0

# Hypocenter coordinates (Km) 
hypox = 0.0;
hypoy = 6.511;
hypoz =-9.463;

print("  ")
print(" START PROGRAM ")
print("  ")

infile = inDir.joinpath(infile)

inData = np.loadtxt(infile, skiprows = 12)

XFinMat = np.reshape(inData[:,0], (ndipin, nstkin))
YFinMat = np.reshape(inData[:,1], (ndipin, nstkin))
ZFinMat = np.reshape(inData[:,2], (ndipin, nstkin))
SlipinMat  = np.reshape(inData[:,4], (ndipin, nstkin))
RiseTinMat = np.reshape(inData[:,5], (ndipin, nstkin))
RupTinMat  = np.reshape(inData[:,6], (ndipin, nstkin))

Slipmax  = max(inData[:,4])
RiseTmax = max(inData[:,5])
RupTmax  = max(inData[:,6])

# Interpolation of fault plane
dstk = np.array([ XFinMat[0,1]-XFinMat[0,0], YFinMat[0,1]-YFinMat[0,0], 
                  ZFinMat[0,1]-ZFinMat[0,0] ])
dstkin = np.linalg.norm(dstk)

ddip = np.array([ XFinMat[1,0]-XFinMat[0,0], YFinMat[1,0]-YFinMat[0,0],
                  ZFinMat[1,0]-ZFinMat[0,0] ])
ddipin = np.linalg.norm(ddip)

dstk = dstk*dhF
ddip = ddip*dhF

# Calculate the strike and dip unitary vetors
univec_stk = np.linalg.norm(dstk)
univec_dip = np.linalg.norm(ddip)

stk = round((nstkin-1)*dstkin)
dip = round((ndipin-1)*ddipin)

print()
print(" Original Fault Dimensions:")
print(f" Strike (Km): {stk} nstk: {nstkin} dstk (Km): {dstkin}" )
print(f" Dip (Km): {dip} ndip: {ndipin} ddip (Km): {ddipin}" )
print(f" Hypocenter Coordinates x, y and z (Km): {hypox}, {hypoy}, {hypoz}" )

dipinVec = np.linspace(0, dip, ndipin)
stkinVec = np.linspace(0, dip, ndipin)
stkinMat, dipinMat = np.meshgrid(stkinVec,dipinVec)

#interpolation
nstk = int(stk/dhF)+1
ndip = int(dip/dhF)+1
stkVec = np.linspace(0, stk, nstk)
dipVec = np.linspace(0, dip, ndip)
stkMat, dipMat = np.meshgrid(stkVec,dipVec)


# Slip Interpolation
SlipF = interpolate.interp2d(stkinVec,dipinVec,SlipinMat, kind = "linear")
SlipMat = SlipF(stkVec, dipVec)
# RiseTime Interpolation
RiseTF = interpolate.interp2d(stkinVec,dipinVec,RiseTinMat, kind = "linear")
RiseTMat = RiseTF(stkVec, dipVec)
# Rupture Time Interpolation
RupTF = interpolate.interp2d(stkinVec,dipinVec,RupTinMat, kind = "linear")
RupTMat = RupTF(stkVec, dipVec)

# Correct limits of interpolated fields
SlipMat[SlipMat<0] = 0
SlipMat[SlipMat>Slipmax] = Slipmax
RiseTMat[RiseTMat<0] = 0
RiseTMat[RiseTMat>RiseTmax] = RiseTmax
RupTMat[RupTMat<0] = 0
RupTMat[RupTMat>RupTmax] = RupTmax

# Coordinates Interpolation
inivec = np.array([ XFinMat[0,0], YFinMat[0,0], ZFinMat[0,0] ])

XFMat = np.zeros((nstk,ndip))
YFMat = np.zeros((nstk,ndip))
ZFMat = np.zeros((nstk,ndip))

for istk in range (0,nstk):
    delta_stk = istk*dstk
    for idip in range (0,ndip ):
        delta_dip = idip*ddip 
        XFMat[idip,istk] = inivec[0] + delta_stk[0] + delta_dip[0]
        YFMat[idip,istk] = inivec[1] + delta_stk[1] + delta_dip[1]
        ZFMat[idip,istk] = inivec[2] + delta_stk[2] + delta_dip[2]

# From matrix to column vector following fortran 
XF3Dzero = XFMat.flatten(order='F').transpose()
YF3Dzero = YFMat.flatten(order='F').transpose()
ZF3Dzero = ZFMat.flatten(order='F').transpose()

IFMat = np.arange(0,XF3Dzero.size).reshape((ndip,nstk),order='F')
ntri  = (nstk-1)*(ndip-1)*2
tri   = np.zeros([ntri,3],dtype=int)  
XY3D  = np.array((XF3Dzero,YF3Dzero)).transpose()

# Delaunay triangulation
tri = Delaunay(XY3D).simplices
ntri = int(tri.size/3)
# jtri = -1
# for istk in range (0,nstk-1):
#     for idip in range (0,ndip-1):
#         jtri += 1
#         tri[jtri,0] = IFMat[idip,istk]
#         tri[jtri,1] = IFMat[idip,istk+1]
#         tri[jtri,2] = IFMat[idip+1,istk+1]
#         jtri += 1
#         tri[jtri,0] = IFMat[idip,istk]
#         tri[jtri,1] = IFMat[idip+1,istk+1]
#         tri[jtri,2] = IFMat[idip+1,istk]
                
triBmarker = np.ones(ntri,)
# Calculate unitary normal, strike and dip vector at each facet
univector = np.zeros((ntri,9))
# Vector normal to earth surface
nsurf = np.array([0,0,-1])          

for itri in range(0,ntri):
    iv0 = tri[itri,0]
    iv1 = tri[itri,1]
    iv2 = tri[itri,2]
    v0 = np.array([ XF3Dzero[iv0], YF3Dzero[iv0], ZF3Dzero[iv0]])
    v1 = np.array([ XF3Dzero[iv1], YF3Dzero[iv1], ZF3Dzero[iv1]])
    v2 = np.array([ XF3Dzero[iv2], YF3Dzero[iv2], ZF3Dzero[iv2]])
    vnormal = np.cross(v1-v0,v2-v0)
    vstrike = np.cross(vnormal,nsurf)
    vdip    = np.cross(vstrike,vnormal)
    univector[itri,0:3] = vnormal/np.linalg.norm(vnormal)
    univector[itri,3:6] = vstrike/np.linalg.norm(vstrike)
    univector[itri,6:9] = vdip/np.linalg.norm(vdip)

fcoor = np.array((XF3Dzero,YF3Dzero,ZF3Dzero)).transpose()

# Add nodes above an below to the fault
XF3Dplus = XF3Dzero+vnormal[0]*dhF
YF3Dplus = YF3Dzero+vnormal[1]*dhF
ZF3Dplus = ZF3Dzero+vnormal[2]*dhF
XF3Dminus = XF3Dzero-vnormal[0]*dhF
YF3Dminus = YF3Dzero-vnormal[1]*dhF
ZF3Dminus = ZF3Dzero-vnormal[2]*dhF

XF3D = np.concatenate((XF3Dzero,XF3Dplus,XF3Dminus),axis=None)
YF3D = np.concatenate((YF3Dzero,YF3Dplus,YF3Dminus),axis=None)
ZF3D = np.concatenate((ZF3Dzero,ZF3Dplus,ZF3Dminus),axis=None)

# Output Dictionary
Data = {}
Data['nstk'] = nstk
Data['ndip'] = ndip
Data['XF3D'] = XF3D
Data['YF3D'] = YF3D
Data['ZF3D'] = ZF3D

print()
print(" Output Fault Dimensions:")
print(f" Strike (Km): {stk} nstk: {nstk} dstk (Km): {np.linalg.norm(dstk)}" )
print(f" Dip (Km): {dip} ndip: {ndip} ddip (Km): {np.linalg.norm(ddip)}" )

name = name+str(int(dhF*1000))

# Write .vector file
fvectorHeader = "%d" %(ntri)
fvector = '../Outputs/3DFaults/'+name+'.vector'
with open(fvector,'wb') as f:
    np.savetxt(f, univector,header=fvectorHeader, comments=' ',fmt='%10.6f')
f.close()

# fileObj = open(ObjName, 'wb') 
# pickle.dump(Fault, fileObj)
# fileObj.close()
pickle.dump(Data, file, protocol=pickle.HIGHEST_PROTOCOL)
# print()
# print(f" Fault strike, dip and normal vector saved in file: {fvector}" )
# print(f" Fault object saved in file: {ObjName}" )

# SlipinColors = Myplots.SetColorbar(SlipinMat)
# SlipColors   = Myplots.SetColorbar(SlipMat)

# mslip=mpl.cm.ScalarMappable(cmap=mpl.cm.viridis)
# mslip.set_array(SlipinVec)
# mriseT=mpl.cm.ScalarMappable(cmap=mpl.cm.viridis)
# mriseT.set_array(RiseTinVec)
# mrupT=mpl.cm.ScalarMappable(cmap=mpl.cm.viridis)
# mrupT.set_array(RupTinVec)

# Slipticks = np.linspace(0, 1.5, 7, endpoint=True)

# fig = plt.figure()
# ax = fig.add_subplot(121, projection='3d')
# ax.plot_surface(XFinMat, YFinMat, ZFinMat, facecolors=SlipinColors, vmin=0, vmax=Slipmax, shade=False)
# ax.set_title("Input Slip")
# ax.set_xlabel(" X (Km)")
# ax.set_ylabel(" Y (Km)")
# ax.set_zlabel(" Z (Km)")
# ax.set_xlim(xmin,xmax)
# ax.set_ylim(ymin,ymax)
# ax.set_zlim(zmin,zmax)
# fig.colorbar(mslip, label='Slip (m)', orientation="horizontal",shrink=.9,ticks=Slipticks)
# ax.azim = -60
# ax.dist = 10
# ax.elev = 10

# ax = fig.add_subplot(122, projection='3d')
# ax.plot_surface(XFMat, YFMat, ZFMat, facecolors=SlipColors, vmin=0, vmax=Slipmax, shade=False)
# ax.set_title("Interpolate Slip")
# ax.set_xlabel(" X (Km)")
# ax.set_ylabel(" Y (Km)")
# ax.set_zlabel(" Z (Km)")
# ax.set_xlim(xmin,xmax)
# ax.set_ylim(ymin,ymax)
# ax.set_zlim(zmin,zmax)
# fig.colorbar(mslip,label='Slip (m)', orientation="horizontal",shrink=.9,ticks=Slipticks)
# ax.azim = -60
# ax.dist = 10
# ax.elev = 10

# fig = plt.figure()
# ax = fig.subplots(2,3)
# ax[0,0].pcolormesh(stkinMat,dipinMat,SlipinMat)
# ax[0,0].axis('equal')
# ax[1,0].pcolormesh(stkMat, dipMat, SlipMat)
# fig.colorbar(mslip,ax=ax[1,0],label='Slip (m)', orientation="horizontal",shrink=.95)
# ax[0,1].pcolormesh(stkinMat,dipinMat,RiseTinMat)
# ax[1,1].pcolormesh(stkMat, dipMat, RiseTMat)
# fig.colorbar(mriseT,ax=ax[1,1],label='Risetime (s)', orientation="horizontal",shrink=.95)
# ax[0,2].pcolormesh(stkinMat,dipinMat,RupTinMat)
# ax[1,2].pcolormesh(stkMat, dipMat, RupTMat)
# fig.colorbar(mrupT,ax=ax[1,2],label='Rupture time (s)', orientation="horizontal",shrink=.95)
# for ax in fig.get_axes():
#     ax.label_outer()

# levels = np.linspace(0,RupTmax,9)

# fig = plt.figure(constrained_layout=True)
# ax1 = fig.add_subplot(121)
# ax1.pcolormesh(stkinMat,dipinMat,SlipinMat)
# cs=ax1.contour(stkinMat,dipinMat,RupTinMat,levels,colors=('w',),linewidths=(0.3,),origin='lower')
# ax1.clabel(cs, fmt='%2.1f', colors='w', fontsize=10)
# fig.colorbar(mslip,ax=ax1,label='Slip (m)', orientation="horizontal",shrink=.95,ticks=Slipticks)
# ax1.set_xlabel(' Strike (Km) ')
# ax1.set_ylabel(' Dip (Km) ')
# ax1.set_xlim([0,stk])
# ax1.set_ylim([0,dip])
# ax1.set_aspect('equal',adjustable='box')
# ax1.set_title(" Input Slip ")

# plt.gca().invert_yaxis()
# ax2 = fig.add_subplot(122)
# ax2.pcolormesh(stkMat,dipMat,SlipMat)
# cs=ax2.contour(stkMat,dipMat,RupTMat,levels,colors=('w',),linewidths=(0.3,),origin='lower')
# ax2.clabel(cs, fmt='%2.1f', colors='w', fontsize=10)
# fig.colorbar(mslip,ax=ax2,label='Slip (m)', orientation="horizontal",shrink=.95,ticks=Slipticks)
# ax2.set_xlabel(' Strike (Km) ')
# ax2.set_xlim([0,stk])
# ax2.set_ylim([0,dip])
# ax2.set_aspect('equal',adjustable='box')
# mpl.pyplot.gca().invert_yaxis()
# ax2.set_title(" Interpolate Slip ")

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(XF3Dzero,YF3Dzero,ZF3Dzero, marker ='.', color='b',label ='Fault Nodes')
# ax.scatter(XF3Dplus,YF3Dplus,ZF3Dplus, marker ='.', color='r', label='ExtraNodes above')
# ax.scatter(XF3Dminus,YF3Dminus,ZF3Dminus, marker ='.', color='g', label="ExtraNodes below")
# ax.set_xlabel(" X (Km)")
# ax.set_ylabel(" Y (Km)")
# ax.set_zlabel(" Z (Km)")
# ax.legend(loc ="upper left")

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_trisurf(XF3D,YF3D,ZF3D, triangles=tri)
# ax.azim = -60
# ax.dist = 10
# ax.elev = 10

print("  ")
print(" END PROGRAM ")
print("  ")



#XFinMat = np.reshape(np.array(list(inData.get('X').values())), (ndipin, nstkin))
# YFinMat = np.reshape(np.array(list(inData.get('Y').values())), (ndipin, nstkin))
# ZFinMat = np.reshape(np.array(list(inData.get('Y').values())), (ndipin, nstkin))

