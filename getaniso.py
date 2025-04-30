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

date = '2025_04_28'
sili = 'SC_37_40'
tricot = '4DFIXNR'

##Pour P = 245 mbar (ip correspond à l'image à cette pression)

ip = 0
if tricot == '4DFIXNR':
#    ip = 104 (essai génie civil)
  ip = 213
if tricot == 'P7NR':
#    ip = 85 (essai génie civil)
  ip = 140
if tricot == 'A1L':
#    ip = 66 (essai génie civil)   
  ip = 78
if tricot == 'P1':
#    ip = 70 (essai génie civil)    
  ip = 120
if tricot == 'P7R':
  ip = 0

ip = -2
X3d = np.loadtxt(fname=f'./{date}/{sili}_{tricot}/results_id/Lx3d.npy', delimiter=' ')[:ip+1]
Y3d = np.loadtxt(fname=f'./{date}/{sili}_{tricot}/results_id/Ly3d.npy', delimiter=' ')[:ip+1]
Z3d = np.loadtxt(fname=f'./{date}/{sili}_{tricot}/results_id/Lz3d.npy', delimiter=' ')[:ip+1]

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


PER = [i for i in np.arange(0.55, 0.95, 0.05)]
Lr = []
for per in PER:
  print(per)
  w = np.where(np.round(Upz)==np.round(per*max(Upz)))
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
  plt.xlim(50,80)
  plt.ylim(0,50)
  plt.show()

plt.scatter(PER, Lr)
plt.show()
np.savetxt(f'./{date}/{sili}_{tricot}/L_aniso.txt', Lr)


def flin(theta, x):
  return theta[0]*x + theta[1]

theta0 = [1, 1]


ax = plt.axes()
ax.set_xlabel('%Uzmax')
ax.set_ylabel('b/a')
inter_PER = np.linspace(0.55, 0.9, 100)


if tricot == '4DFIXNR':
  Lr4DFIXNR = Lr
  def fun4DFIXNR(theta):
    return flin(theta, np.array(PER)) - np.array(Lr4DFIXNR)

  res4DFIXNR = least_squares(fun4DFIXNR, theta0)
  plt.scatter(PER, Lr4DFIXNR, label=f'4DFIXNR_pente={np.round(res4DFIXNR.x[0],3)}', c='b')
  plt.plot(inter_PER, flin(res4DFIXNR.x, inter_PER), c='b')

#tricot = 'P7NR'
if tricot == 'P7NR':
  LrP7NR = Lr
  def funP7NR(theta):
    return flin(theta, np.array(PER)) - np.array(LrP7NR)

  resP7NR = least_squares(funP7NR, theta0)
  plt.scatter(PER, LrP7NR, label=f'P7NR_pente={np.round(resP7NR.x[0],3)}', c='orange')
  plt.plot(inter_PER, flin(resP7NR.x, inter_PER), c='orange')

#tricot = 'A1L'
if tricot == 'A1L':
  LrA1L = Lr
  def funA1L(theta):
    return flin(theta, np.array(PER)) - np.array(LrA1L)

  resA1L = least_squares(funA1L, theta0)
  plt.scatter(PER, LrA1L, label=f'A1L_pente={np.round(resA1L.x[0],3)}', c='g')
  plt.plot(inter_PER, flin(resA1L.x, inter_PER), c='g')

#tricot = 'P1'
if tricot == 'P1':
  LrP1 = Lr
  def funP1(theta):
    return flin(theta, np.array(PER)) - np.array(LrP1)

  resP1 = least_squares(funP1, theta0)
  plt.scatter(PER, LrP1, label=f'P1_pente={np.round(resP1.x[0],3)}', c='r')
  plt.plot(inter_PER, flin(resP1.x, inter_PER), c='r')

  

plt.legend()
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


