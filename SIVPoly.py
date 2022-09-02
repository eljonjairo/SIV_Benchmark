#!/home/jon/anaconda3/bin/python
#
# Generate .Poly and .var file for tetgen
#
# John Diaz July 2022

# To Do List:
#  

import os
import warnings

import numpy as np
import pickle
import math 
import matplotlib.pyplot as plt
import FDealunay

from scipy.spatial import Delaunay
from scipy.spatial import distance
from pathlib import Path
from IPython import get_ipython

get_ipython().magic('reset -sf')
warnings.filterwarnings("ignore")
plt.close('all') 
os.system('clear')

# Name of the Fault object
ObjName = 'Outputs/3DFaults/SIVBenchmarkDsef0_dhF500m.pickle'

# Output Poly folder
PolyFolder = Path('Outputs/Poly')

# Km to m
m = 1000
#Horizontal and Topography steps (Km)
dh = 4;
dhTopo = 4;

# Distance to remove points close to the fault (m)
distToFault = 1.0;

#Z Coordinates of Velocity Model Layers (Km) (maximum 4 km in between)
# Velocity layers in z: 2.0, 4.8, 18.0, 24.0, 40.0
Zlayer = np.array([ 2.0, 4.8, 6.0, 10.0, 14.0, 18.0, 24.0, 28.0, 32.0, 36.0, 40.0 ])     # Last is the limit of the Domain 

# Domain Limits (Km) 
#SIV Benchmark
xUTM_ini  = -50;
xUTM_end  =  50;
yUTM_ini  = -50;
yUTM_end  =  50;

print("  ")
print(" START PROGRAM ")
print("  ")

# Load the Fault object
with open(ObjName, 'rb') as handle:
    Fault = pickle.load(handle)

nstk = Fault['nstk']
ndip = Fault['ndip']

XF3D = np.array(Fault['XF3D'])
YF3D = np.array(Fault['YF3D'])
ZF3D = np.array(Fault['ZF3D'])
XF3Dadd = np.array(Fault['XF3Dadd'])
YF3Dadd = np.array(Fault['YF3Dadd'])
ZF3Dadd = np.array(Fault['ZF3Dadd'])
dhFault = Fault['dhF']
nzlayer = Zlayer.size

print()
print(' Build Topography ...  ')

xUTM_end = xUTM_end+dhTopo
yUTM_end = yUTM_end+dhTopo
xTopo = np.arange(xUTM_ini,xUTM_end,dhTopo)
yTopo = np.arange(yUTM_ini,yUTM_end,dhTopo)

xTopoMat, yTopoMat = np.meshgrid(xTopo, yTopo, indexing='xy')

xTopoVec = xTopoMat.flatten(order='F')
yTopoVec = yTopoMat.flatten(order='F')
zTopoVec = np.zeros(xTopoVec.size,)

Xvector = xTopoVec
Yvector = yTopoVec
Zvector = zTopoVec

print(' Dealunay Triangulation of Topography')
print()
# Topo Delaunay triangulation
XYTopo = np.array((xTopoVec,yTopoVec)).transpose()
triTopo = Delaunay(XYTopo).simplices

ntri = int(triTopo.size/3)
triB = np.zeros(ntri,)

triGlobal = triTopo
triBmarker = triB

print(" Build 3D Model ...")
print()

x = np.arange(xUTM_ini,xUTM_end,dh)
y = np.arange(yUTM_ini,yUTM_end,dh)

xMat,yMat = np.meshgrid(x, y, indexing='xy')
xLayerVec = xMat.flatten(order='F')
yLayerVec = yMat.flatten(order='F')

print(" Horizontal Layers at z: ")
for izl in Zlayer:
    print(f" {izl} Km." )
    zLayerVec = -np.ones(xLayerVec.size)*izl
    Xvector = np.append(Xvector,xLayerVec)
    Yvector = np.append(Yvector,yLayerVec)
    Zvector = np.append(Zvector,zLayerVec)

print()
print(" Adding Fault Nodes ")

fcoor = np.array(Fault['fcoor'])

# Remove nodes closer to the fault
for icoor in fcoor:
    XYZvec = np.array([Xvector,Yvector,Zvector]).transpose()
    nx = int(XYZvec.size/3)
    dist = np.zeros(nx,)
    for ix in range(0,nx):
        dist[ix] = distance.euclidean(XYZvec[ix],icoor)
    Xvector = np.delete(Xvector,np.where(dist<distToFault))    
    Yvector = np.delete(Yvector,np.where(dist<distToFault))
    Zvector = np.delete(Zvector,np.where(dist<distToFault))

print(' Adding Fault Facets')
# Add Fault nodes triangulation, triBmarker and Fault Nodes   
triF  = np.array(Fault['tri']) + len(Xvector)
ntriF = Fault['ntri']
triBmarkerF = Fault['triBmarker']

triGlobal = np.concatenate((triGlobal,triF))
triBmarker = np.concatenate((triBmarker,triBmarkerF))
Xvector = np.concatenate((Xvector,XF3D))
Yvector = np.concatenate((Yvector,YF3D))
Zvector = np.concatenate((Zvector,ZF3D))

#Xvector = np.concatenate((Xvector,XF3D,XF3Dadd))
#Yvector = np.concatenate((Yvector,YF3D,YF3Dadd))
#Zvector = np.concatenate((Zvector,ZF3D,ZF3Dadd))

print()
print(" Dealunay triangulation of borders...")
# -------------------------------------------------------------------------
# Facet section
# Building Facets
# Delaunay Triangulation of the boundaries

xmin = min(Xvector)
xmax = max(Xvector)
ymin = min(Yvector)
ymax = max(Yvector)
zmin = min(Zvector)
zmax = max(Zvector)

# Facets of the west boundary nodes
triW = FDealunay.Facetx(Xvector,Yvector,Zvector,xmin)   
triGlobal = np.concatenate((triGlobal,triW))
triB = np.zeros(int(triW.size/3),)
triBmarker = np.concatenate((triBmarker,triB))
print(f' East Facets at x: {xmin} km')

# Facets of the east boundary nodes
triE = FDealunay.Facetx(Xvector,Yvector,Zvector,xmax)
triGlobal = np.concatenate((triGlobal,triE))
triB = np.zeros(int(triE.size/3),)
triBmarker = np.concatenate((triBmarker,triB))
print(f' West Facets at x: {xmax} km')

# Facets of the north boundary nodes
triN = FDealunay.Facety(Xvector,Yvector,Zvector,ymax)
triGlobal = np.concatenate((triGlobal,triN))
triB = np.zeros(int(triN.size/3),)
triBmarker = np.concatenate((triBmarker,triB))
triPlot = triGlobal
print(f' North Facets at y: {ymax} km')

# Facets of the south boundary nodes
triS = FDealunay.Facety(Xvector,Yvector,Zvector,ymin)
triGlobal = np.concatenate((triGlobal,triS))
triB = np.zeros(int(triS.size/3),)
triBmarker = np.concatenate((triBmarker,triB))
print(f' South Facets at y: {ymin} km')

# Facets of the bottom boundary nodes
triD = FDealunay.Facetz(Xvector,Yvector,Zvector,zmin)
triGlobal = np.concatenate((triGlobal,triD))
triPlot = np.concatenate((triPlot,triD))
triB = np.zeros(int(triD.size/3),)
triBmarker = np.concatenate((triBmarker,triB))
print(f' Bottom Facets at z: {zmin} km')

print()
print(" Borders of Computational Domain, min and max values in x y and z in Km:")
print(f" {xmin} {xmax} {ymin} {ymax} {zmin} {zmax} ")

print()
print(" Writing output files...")

outname = Fault['outname']
# Write var file
inradiusF = dhFault*1000*math.sqrt(3)/6;   # inradius of triangle assuming it is equilateral
area = ((inradiusF*math.sqrt(24))**2)*(math.sqrt(3)/4)
fvar = outname+'.var'
with open (PolyFolder.joinpath(fvar),'w') as f:
      f.write("# Facet Constrains \n")
      f.write("%d \n" %(ntriF))
      for itri in range(0,ntriF):
          f.write("%6d %6d %12.3f #Set Maximum area on Facets (1) \n"\
                  %(itri+1,itri+1,area))
              
      f.write("# Segment Constraints \n")        
          
      f.write(" 0 # No Constrains")    
f.close()     
print()
print(f' {fvar} file created ...')

# Writing Limits for refine procedure (see run folder)
fLim = outname+'.lim'
with open (PolyFolder.joinpath(fLim),'w') as f:
      f.write("Limits of Poly file: xmin xmax ymin ymax zmin zmax in m \n")
      f.write(' %12.3f %12.3f %12.3f %12.3f %12.3f %12.3f \n' \
              %(xmin*m,xmax*m,ymin*m,ymax*m,zmin*m,zmax*m))
f.close()     
print(f' {fLim} file created ...')

# Writing Poly file
nx   = Xvector.size
ntri = int(triGlobal.size/3)
fPoly = outname+'.poly'
with open (PolyFolder.joinpath(fPoly),'w') as f:
      f.write("# Part 1 - node list \n")
      f.write("# node count, 3 dim, no attribute, no boundary maker \n")
      f.write(' %d %d %d %d  \n' %(nx,3,0,0))
      f.write('# Node index, node coordinates \n')
      for ix in range(0,nx):
          f.write(' %6d %12.3f %12.3f %12.3f \n'\
                  %(ix+1,Xvector[ix]*m,Yvector[ix]*m,Zvector[ix]*m) )
      f.write('# Part 2 - facet list \n')
      f.write('# facet count, boundary marker \n')
      f.write('  %d  %d \n' %(ntri,1))
      f.write('# Factes \n')
      for itri in range(0,ntri):
          f.write(' %3d %3d %3d # 1 polygon, no hole, boundary marker \n' \
                  %(1,0,triBmarker[itri]))
          f.write(' %3d %8d %8d %8d \n' \
                  %(3,triGlobal[itri,0]+1,triGlobal[itri,1]+1,triGlobal[itri,2]+1))
      f.write('# Part 3 - hole list \n')
      f.write('0        # no hole \n')
      f.write('# Part 4 - region list \n')
      f.write('0        # no region \n')
f.close()     
print(f' {fPoly} file created ...')     

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(Xvector,Yvector,Zvector, marker ='.')
ax.set_xlabel(" X (Km)")
ax.set_ylabel(" Y (Km)")
ax.set_zlabel(" Z (Km)")


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(Xvector,Yvector,Zvector, triangles=triGlobal)
ax.azim = -60
ax.dist = 10
ax.elev = 10
ax.set_title("Global")


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(Xvector,Yvector,Zvector, triangles=triPlot)
ax.azim = -60
ax.dist = 10
ax.elev = 10
ax.set_title("Global")

print("  ")
print(" END PROGRAM ")
print("  ")

   
    