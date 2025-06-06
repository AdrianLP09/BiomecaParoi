import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import Rbf
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt


def fit_ellipse(x, y):
    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()

##Pour P = 245 mbar (ip correspond à l'image à cette pression)
def getaniso(tricot: str):
  ip = 0
  if tricot == '4DFIXNR':
    ip = 104
  if tricot == 'P7NR':
    ip = 85
  if tricot == 'A1L':
    ip = 66    
  if tricot == 'P1':
    ip = 70     


  date = "2023_07_06"
  #X3d = np.loadtxt(fname=f'../data/SC37_40_{tricot}/X3d_SC37_40_{tricot}.txt', delimiter=' ')[:ip+1]
  #Y3d = np.loadtxt(fname=f'../data/SC37_40_{tricot}/Y3d_SC37_40_{tricot}.txt', delimiter=' ')[:ip+1]
  #Z3d = np.loadtxt(fname=f'../data/SC37_40_{tricot}/Z3d_SC37_40_{tricot}.txt', delimiter=' ')[:ip+1]

  X3d = np.loadtxt(fname=f'../opti_angle_calib/{date}/40d_cd/X3d_SC37_40.txt', delimiter=' ')[:ip+1]
  Y3d = np.loadtxt(fname=f'../opti_angle_calib/{date}/40d_cd/Y3d_SC37_40.txt', delimiter=' ')[:ip+1]
  Z3d = np.loadtxt(fname=f'../opti_angle_calib/{date}/40d_cd/Z3d_SC37_40.txt', delimiter=' ')[:ip+1]

  X0min = min(X3d[0])
  X0max = max(X3d[0])
  Y0min = min(Y3d[0])
  Y0max = max(Y3d[0])

  cx = (X0max-X0min)/2 + X0min
  cy = (Y0max-Y0min)/2 + Y0min

  Xmesh, Ymesh = np.meshgrid(np.linspace(X0min, X0max, 200), np.linspace(Y0min, Y0max, 200), indexing='xy')
  Ymesh = np.flip(Ymesh)

  #Fonctions interpolatrices
  interpfunction = 'linear'

  Rbfx = []
  Rbfy = []
  Rbfz = []

  for i in range(len(X3d)):
    Rbfx.append(Rbf(X3d[0], Y3d[0], X3d[i], function=interpfunction))
    Rbfy.append(Rbf(X3d[0], Y3d[0], Y3d[i], function=interpfunction))
    Rbfz.append(Rbf(X3d[0], Y3d[0], Z3d[i], function=interpfunction))
  
  
  Xmeshc = Xmesh.copy()
  Ymeshc = Ymesh.copy()
  c=0
  a = np.where((Xmesh-Xmesh[100][100])**2 + (Ymesh-Ymesh[100][100])**2 > ((X0max-X0min)/2)**2)
  for i in range(len(Xmesh)):
    for j in range(len(Ymesh)):
      if ((Xmesh[i][j]-Xmesh[100][100])**2 + (Ymesh[i][j]-Ymesh[100][100])**2) > ((X0max-X0min)/2)**2:
        Xmeshc[i][j] = 'nan'
        Ymeshc[i][j] = 'nan'
        c+=1
  
  XXc = []
  YYc = []
  ZZc = []

  for i in range(len(Rbfx)):
    XXc.append(Rbfx[i](Xmeshc, Ymeshc))
    YYc.append(Rbfy[i](Xmeshc, Ymeshc))
    ZZc.append(Rbfz[i](Xmeshc, Ymeshc))

  #Calcul des vecteurs déplacements  
  Uxc = []
  Uyc = []
  Uzc = []

  for i in range(len(XXc)):
    Uxc.append(XXc[i] - XXc[0])
    Uyc.append(YYc[i] - YYc[0])
    Uzc.append(ZZc[i] - ZZc[0])

  x0 = XXc[0][~np.isnan(XXc[0])]
  y0 = YYc[0][~np.isnan(YYc[0])]
  z0 = ZZc[0][~np.isnan(ZZc[0])]
  xp = XXc[ip][~np.isnan(XXc[ip])]
  yp = YYc[ip][~np.isnan(YYc[ip])]
  zp = ZZc[ip][~np.isnan(ZZc[ip])]

  Upx = xp-x0
  Upy = yp-y0
  Upz = zp-z0


  PER = [i for i in np.arange(0.1, 1.05, 0.05)]
  Lr = []
  for per in PER:
    print(per)
    w = np.where(np.round(Upz,1)==np.round(per*max(Upz),1))
    res = fit_ellipse(xp[w], yp[w])
    a = (-np.sqrt(2*(res[0]*res[4]**2 + res[2]*res[3]**2 - res[1]*res[3]*res[4] + (res[1]**2 - 4*res[0]*res[2])*res[5])*((res[0]+res[2]) + np.sqrt((res[0]-res[2])**2 + res[1]**2))))/(res[1]**2 - 4*res[0]*res[2])
    b = (-np.sqrt(2*(res[0]*res[4]**2 + res[2]*res[3]**2 - res[1]*res[3]*res[4] + (res[1]**2 - 4*res[0]*res[2])*res[5])*((res[0]+res[2]) - np.sqrt((res[0]-res[2])**2 + res[1]**2))))/(res[1]**2 - 4*res[0]*res[2])
    x0 = (2*res[2]*res[3] - res[1]*res[4])/(res[1]**2 - 4*res[0]*res[2])
    y0 = (2*res[0]*res[4] - res[1]*res[3])/(res[1]**2 - 4*res[0]*res[2])
    teh = np.arctan((res[2] - res[0] - np.sqrt((res[0] - res[2])**2 + res[1]**2))/res[1])
    print('aniso:', min(b/a, a/b))
    print('angle:', teh*180/np.pi)
    Lr.append(min(b/a, a/b))
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    e = Ellipse(xy = [x0, y0], width = 2*a, height = 2*b, angle = 180*teh/np.pi, facecolor='white', edgecolor='b', linewidth=12)
    ax.add_artist(e)
    plt.scatter(xp[w], yp[w], c='r', linewidths=0.02)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    plt.show()

  plt.scatter(PER, Lr)
  plt.show()



  x = X3d[70]
  y = Y3d[70]
  z = Z3d[70]
  x1 = [cx+b*np.sin(teh), cx-b*np.sin(teh)]
  y1 = [cy-b*np.cos(teh), cy+b*np.cos(teh)]
  z1 = [76,76]
  x2 = [cx+a*np.cos(teh), cx-a*np.cos(teh)]
  y2 = [cy+a*np.sin(teh), cy-a*np.sin(teh)]
  z2 = [76,76]

  fig = plt.figure(figsize = (16, 9))
  ax = plt.axes(projection ="3d")
  ax.grid(visible = True, color ='grey',
    linestyle ='-.', linewidth = 0.3,
    alpha = 0.2)
  my_cmap = plt.get_cmap('hsv')
  sctt = ax.scatter3D(x, y, z,
            alpha = 0.8,
            c = z,
            cmap = my_cmap)
  ax.plot(x1, y1, z1, linewidth=5, color='k')
  ax.plot(x2, y2, z2, linewidth=5, color='k')
  plt.title("Results P1")
  ax.set_xlabel('x (mm)', fontweight ='bold')
  ax.set_ylabel('y (mm)', fontweight ='bold')
  ax.set_zlabel('z (mm)', fontweight ='bold')
  fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
  plt.show()


  return(Lr) 

Lr4d = getaniso('4DFIXNR')
#Lrp7 = getaniso('P7NR')
#Lra1 = getaniso('A1L')
#Lrp1 = getaniso('P1')
PER = [i for i in np.arange(0.5, 0.95, 0.05)]

ax = plt.axes()
ax.set_xlabel('%Uzmax')
ax.set_ylabel('b/a')
plt.scatter(PER, Lr4d, label='4DFIXNR')
#plt.scatter(PER, Lrp7, label='P7NR')
#plt.scatter(PER, Lra1, label='A1L')
#plt.scatter(PER, Lrp1, label='P1')
plt.legend()
plt.show()


#Courbes de tendance pour les tricots P7NR et P1
#def fp7(theta, x):
  #return(theta[0]*x + theta[1])
  
#def funp7(theta):
  #return (fp7(theta, np.array(PER[:-1])) - np.array(Lrp7[:-1]))
  
#theta0p7 = [1,1]
#resp7 = least_squares(funp7, theta0p7)

#def fp1(theta, x):
  #return(theta[0]*x**4 + theta[1]*x**3 + theta[2]*x**2 + theta[3]*x + theta[4])
  
#def funp1(theta):
  #return (fp1(theta, np.array(PER[:-1])) - np.array(Lrp1[:-1]))
  
#theta0p1 = [1,1,1,1,1]
#resp1 = least_squares(funp1, theta0p1)

#inter_per = np.linspace(0.5,0.9,100)
#plt.plot(inter_per, fp1(resp1.x, inter_per))
#plt.show()



#x = X3d[70]
#y = Y3d[70]
#z = Z3d[70]
#x1 = [cx+b*np.sin(teh), cx-b*np.sin(teh)]
#y1 = [cy-b*np.cos(teh), cy+b*np.cos(teh)]
#z1 = [76,76]
#x2 = [cx+a*np.cos(teh), cx-a*np.cos(teh)]
#y2 = [cy+a*np.sin(teh), cy-a*np.sin(teh)]
#z2 = [76,76]

#fig = plt.figure(figsize = (16, 9))
#ax = plt.axes(projection ="3d")
#ax.grid(visible = True, color ='grey',
   #linestyle ='-.', linewidth = 0.3,
   #alpha = 0.2)
#my_cmap = plt.get_cmap('hsv')
#sctt = ax.scatter3D(x, y, z,
		  #alpha = 0.8,
		  #c = z,
		  #cmap = my_cmap)
#ax.plot(x1, y1, z1, linewidth=5, color='k')
#ax.plot(x2, y2, z2, linewidth=5, color='k')
#plt.title("Results P1")
#ax.set_xlabel('x (mm)', fontweight ='bold')
#ax.set_ylabel('y (mm)', fontweight ='bold')
#ax.set_zlabel('z (mm)', fontweight ='bold')
#fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
#plt.show()


