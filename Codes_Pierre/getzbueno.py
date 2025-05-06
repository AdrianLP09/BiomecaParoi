import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.interpolate import Rbf


X3d = np.loadtxt(fname='./2023_07_06/40d_cd/X3d.txt', delimiter=' ')
Y3d = np.loadtxt(fname='./2023_07_06/40d_cd/Y3d.txt', delimiter=' ')
Z3d = np.loadtxt(fname='./2023_07_06/40d_cd/Z3d.txt', delimiter=' ')

Limage = sorted(glob('./2023_07_06/40d_cd/left/' + '0*'))
Ltime= []
for i in range(len(Limage)):
  Ltime.append(float(Limage[i].split('/')[4].split('_')[1].split('.tiff')[0]))
t_mot = np.loadtxt('./2023_07_06/40d_cd/data_pos_mot', skiprows=1, delimiter=',')[:,1]
z_mot = np.loadtxt('./2023_07_06/40d_cd/data_pos_mot', skiprows=1, delimiter=',')[:,0]
f_mot = Rbf(t_mot, z_mot)

Z3d_mot = f_mot(Ltime)

Zmoy = []
Zec = []
for i in range(len(Z3d)):
  Zmoy.append(np.mean(Z3d[i]))
  Zec.append(np.std(Z3d[i]))
  
zbueno = Z3d_mot[np.where(Zec == min(Zec))]
print(zbueno)

