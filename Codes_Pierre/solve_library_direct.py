import sigfig as sgf
import numpy as np
from matplotlib import pyplot as plt

 
class Direct_Polynome(dict) :
    def __init__(self, _dict_):
        self._dict_ = _dict_
        self.polynomial_form = _dict_['polynomial_form']

    def pol_form (self, Xl, Xr) :
        """Create the matrix M = f(Xl,Xr) with f the polynomial function of 
        degree n
        
        Args:
           Xl : numpy.ndarray
               Left detected points  Xl(Xl1, Xl2)
           Xr : numpy.ndarray
               Right detected points  Xr(Xr1, Xr2)
               
        Returns:
           M : numpy.ndarray
               M = f(Xl,Xr)
        """
        polynomial_form = self.polynomial_form
        Xl1, Xl2 = Xl
        Xr1, Xr2 = Xr
        
        n = len(Xl1)
        if   polynomial_form == 1 :
            M = np.asarray ([np.ones((n)),   Xl1,           Xl2,            Xr1,            Xr2])

        elif polynomial_form == 2 :
            Xl12 = Xl1 * Xl1
            Xl22 = Xl2 * Xl2
            Xr12 = Xr1 * Xr1
            Xr22 = Xr2 * Xr2
            M = np.asarray ([np.ones((n)),   Xl1,           Xl2,            Xr1,            Xr2,
                             Xl12,           Xl1*Xl2,       Xl1*Xr1,        Xl1*Xr2,        Xl22,
                             Xl2*Xr1,        Xl2*Xr2,       Xr12,           Xr1*Xr2,        Xr22])

        elif polynomial_form == 3 :
            Xl12 = Xl1 * Xl1
            Xl13 = Xl1 * Xl1 * Xl1
            Xl22 = Xl2 * Xl2
            Xl23 = Xl2 * Xl2 * Xl2
            Xr12 = Xr1 * Xr1
            Xr13 = Xr1 * Xr1 * Xr1
            Xr22 = Xr2 * Xr2
            Xr23 = Xr2 * Xr2 * Xr2
            M = np.asarray ([np.ones((n)),   Xl1,           Xl2,            Xr1,            Xr2,
                             Xl12,           Xl1*Xl2,       Xl1*Xr1,        Xl1*Xr2,        Xl22,
                             Xl2*Xr1,        Xl2*Xr2,       Xr12,           Xr1*Xr2,        Xr22,
                             Xl13,           Xl12*Xl2,      Xl12*Xr1,       Xl12*Xr2,       Xl1*Xl22,
                             Xl1*Xl2*Xr1,    Xl1*Xl2*Xr2,   Xl1*Xr12,       Xl1*Xr1*Xr2,    Xl1*Xr22,
                             Xl23,           Xl22*Xr1,      Xl22*Xr2,       Xl2*Xr12,       Xl2*Xr1*Xr2,    
                             Xl2*Xr22,       Xr13,          Xr12*Xr2,       Xr1*Xr22,       Xr23])

        elif polynomial_form == 4 :
            Xl12 = Xl1 * Xl1
            Xl13 = Xl1 * Xl1 * Xl1
            Xl14 = Xl1 * Xl1 * Xl1 * Xl1
            Xl22 = Xl2 * Xl2
            Xl23 = Xl2 * Xl2 * Xl2
            Xl24 = Xl2 * Xl2 * Xl2 * Xl2
            Xr12 = Xr1 * Xr1
            Xr13 = Xr1 * Xr1 * Xr1
            Xr14 = Xr1 * Xr1 * Xr1 * Xr1
            Xr22 = Xr2 * Xr2
            Xr23 = Xr2 * Xr2 * Xr2
            Xr24 = Xr2 * Xr2 * Xr2 * Xr2
            M = np.asarray ([np.ones((n)),   Xl1,           Xl2,            Xr1,            Xr2,
                             Xl12,           Xl1*Xl2,       Xl1*Xr1,        Xl1*Xr2,        Xl22,
                             Xl2*Xr1,        Xl2*Xr2,       Xr12,           Xr1*Xr2,        Xr22,
                             Xl13,           Xl12*Xl2,      Xl12*Xr1,       Xl12*Xr2,       Xl1*Xl22,
                             Xl1*Xl2*Xr1,    Xl1*Xl2*Xr2,   Xl1*Xr12,       Xl1*Xr1*Xr2,    Xl1*Xr22,
                             Xl23,           Xl22*Xr1,      Xl22*Xr2,       Xl2*Xr12,       Xl2*Xr1*Xr2,    
                             Xl2*Xr22,       Xr13,          Xr12*Xr2,       Xr1*Xr22,       Xr23,
                             Xl14,           Xl13*Xl2,      Xl13*Xr1,       Xl13*Xr2,       Xl12*Xl22,
                             Xl12*Xl2*Xr1,   Xl12*Xl2*Xr2,  Xl12*Xr12,      Xl12*Xr1*Xr2,   Xl12*Xr22,
                             Xl1*Xl23,       Xl1*Xl22*Xr1,  Xl1*Xl22*Xr2,   Xl1*Xl2*Xr12,   Xl1*Xl2*Xr1*Xr2,
                             Xl1*Xl2*Xr22,   Xl1*Xr13,      Xl1*Xr12*Xr2,   Xl1*Xr1*Xr22,   Xl1*Xr23,
                             Xl24,           Xl23*Xr1,      Xl23*Xr2,       Xl22*Xr12,      Xl22*Xr1*Xr2,
                             Xl22*Xr22,      Xl2*Xr13,      Xl2*Xr12*Xr2,   Xl2*Xr1*Xr22,   Xl2*Xr23,
                             Xr14,           Xr13*Xr2,      Xr12*Xr22,      Xr1*Xr23,       Xr24])

        return (M)
    

def fit_plan_to_points(point,
                       title = 'no title'):
    """Plot the median plan from a serie of points
    
    Args:
       point : numpy.ndarray (shape = m,3)
           Real points x(x1, x2, x3)       
       title : str
           Title of the plotted figure
            
    Returns:
       plot points + associated plan
    """
    xs, ys, zs = point 
    ax = plt.subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, color='b')
    
    # do fit
    tmp_A = []
    tmp_b = []
    for i in range(len(xs)):
        tmp_A.append([xs[i], ys[i], 1])
        tmp_b.append(zs[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    
    # Manual solution
    fit = (A.T * A).I * A.T * b
    errors = b - A * fit
    residual = np.linalg.norm(errors)
    mean_error = np.mean (abs(errors))
    errors = np.reshape(errors, (len(errors)))
    
    # plot plan
    X,Y = np.meshgrid(np.linspace(np.min(xs), np.max(xs), 10),
                      np.linspace(np.min(ys), np.max(ys), 10))
    Z = np.zeros(X.shape)
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
    ax.plot_wireframe(X,Y,Z, color='k')
        
    ax.set_title(title)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    
    fit = np.transpose(np.array(fit))[0]
    
    return (fit, errors, mean_error, residual)

def fit_plans_to_points(points, 
                        title = 'no title'):
    """Plot the medians plans from series of points
    
    Args:
       points : numpy.ndarray (shape = l,m,3)
           Real points x(x1, x2, x3)       
       title : str
           Title of the plotted figures
            
    Returns:
       plot points + associated plans
    """
    # plot raw data
    l = len (points)
    fit = np.zeros((l, 3))
    errors = []
    mean_error = np.zeros(l)
    residual = np.zeros(l)
    maxerror = []
    for i in range (len(points)) :
        point = points[i]
        fit[i], errori, mean_error[i], residual[i] = fit_plan_to_points(point, 
                                                                        title = title)
        maxerror.append(np.max(abs(errori)))
        errors.append(errori)
    plt.figure()
    plt.show()    
    print('Plan square max error = ', sgf.round((max(maxerror)), sigfigs =3), ' mm')
    print('Plan square mean error = ', sgf.round((np.mean(mean_error**2))**(1/2), sigfigs = 3), ' mm')
    print('Plan square mean residual = ', sgf.round((np.mean(residual**2))**(1/2), sigfigs = 3))

    return (fit, errors, mean_error, residual)

def refplans(xc1, x3_list) :
    """Plot the medians plans from references points
    
    Args:
       xc1 : numpy.ndarray (shape = 3,n)
           Real points x(x1, x2, x3)       
       x3_list : numpy.ndarray
           List of the different plans coordinates
            
    Returns:
       plot points + associated plans
    """
    m, n = xc1.shape
    x,y,z = xc1
    xcons = []
    p0, pf = 0, 0
    for z_i in x3_list :
        while z[pf] == z_i :
            pf += 1
            if pf > n-1 :
                break
        plan = np.array ([x[p0:pf], y[p0:pf], z[p0:pf]])
        p0 = pf
        xcons.append (plan)

    fit_plans_to_points(xcons, 
                        title = 'Calibration plans')

