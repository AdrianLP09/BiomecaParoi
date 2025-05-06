import crappy
import numpy as np


def r_to_l_trans(r_pix, matrix):

  l_pix = []
  for i in range(len(r_pix)):
    vect_r = list(r_pix[i])
    vect_r.append(1)
    vect_l = np.dot(matrix, vect_r)
    vect_l = list(vect_l)
    vect_l[0] = vect_l[0] / vect_l[2]
    vect_l[1] = vect_l[1] / vect_l[2]
    del vect_l[2]
    l_pix.append(vect_l)
  return np.array(l_pix)


def get_all_pix(all_pxl, all_pyl, all_pxr, all_pyr):

  all_pix = np.zeros((2, len(all_pxl), 2))
  for i in range(len(all_pxl)):
    all_pix[0][i][0] = all_pyl[i]
    all_pix[0][i][1] = all_pxl[i]
    all_pix[1][i][0] = all_pyr[i]
    all_pix[1][i][1] = all_pxr[i]
  return all_pix


class DirectPolynom(dict):

  def __init__(self, _dict_):

    super().__init__()
    self._dict_ = _dict_
    self.polynomial_form = _dict_['polynomial_form']

  def pol_form(self, xl, xr):
    """Create the matrix M = f(Xl,Xr) with f the polynomial function of
    degree n

    Args:
       xl : numpy.ndarray
           Left detected points  Xl(Xl1, Xl2)
       xr : numpy.ndarray
           Right detected points  Xr(Xr1, Xr2)

    Returns:
       mat : numpy.ndarray
           M = f(Xl,Xr)
    """

    polynomial_form = self.polynomial_form
    xl1, xl2 = xl
    xr1, xr2 = xr

    n = len(xl1)
    if polynomial_form == 1:
      mat = np.asarray([np.ones(n), xl1, xl2, xr1, xr2])

    elif polynomial_form == 2:
      xl12 = xl1 * xl1
      xl22 = xl2 * xl2
      xr12 = xr1 * xr1
      xr22 = xr2 * xr2
      mat = np.asarray([np.ones(n), xl1, xl2, xr1, xr2,
                        xl12, xl1 * xl2, xl1 * xr1, xl1 * xr2, xl22,
                        xl2 * xr1, xl2 * xr2, xr12, xr1 * xr2, xr22])

    elif polynomial_form == 3:
      xl12 = xl1 * xl1
      xl13 = xl1 * xl1 * xl1
      xl22 = xl2 * xl2
      xl23 = xl2 * xl2 * xl2
      xr12 = xr1 * xr1
      xr13 = xr1 * xr1 * xr1
      xr22 = xr2 * xr2
      xr23 = xr2 * xr2 * xr2
      mat = np.asarray([np.ones(n), xl1, xl2, xr1, xr2,
                        xl12, xl1 * xl2, xl1 * xr1, xl1 * xr2, xl22,
                        xl2 * xr1, xl2 * xr2, xr12, xr1 * xr2, xr22,
                        xl13, xl12 * xl2, xl12 * xr1, xl12 * xr2, xl1 * xl22,
                        xl1 * xl2 * xr1, xl1 * xl2 * xr2, xl1 * xr12,
                        xl1 * xr1 * xr2, xl1 * xr22,
                        xl23, xl22 * xr1, xl22 * xr2, xl2 * xr12,
                        xl2 * xr1 * xr2,
                        xl2 * xr22, xr13, xr12 * xr2, xr1 * xr22, xr23])

    elif polynomial_form == 4:
      xl12 = xl1 * xl1
      xl13 = xl1 * xl1 * xl1
      xl14 = xl1 * xl1 * xl1 * xl1
      xl22 = xl2 * xl2
      xl23 = xl2 * xl2 * xl2
      xl24 = xl2 * xl2 * xl2 * xl2
      xr12 = xr1 * xr1
      xr13 = xr1 * xr1 * xr1
      xr14 = xr1 * xr1 * xr1 * xr1
      xr22 = xr2 * xr2
      xr23 = xr2 * xr2 * xr2
      xr24 = xr2 * xr2 * xr2 * xr2
      mat = np.asarray([np.ones(n), xl1, xl2, xr1, xr2,
                        xl12, xl1 * xl2, xl1 * xr1, xl1 * xr2, xl22,
                        xl2 * xr1, xl2 * xr2, xr12, xr1 * xr2, xr22,
                        xl13, xl12 * xl2, xl12 * xr1, xl12 * xr2, xl1 * xl22,
                        xl1 * xl2 * xr1, xl1 * xl2 * xr2, xl1 * xr12,
                        xl1 * xr1 * xr2, xl1 * xr22,
                        xl23, xl22 * xr1, xl22 * xr2, xl2 * xr12,
                        xl2 * xr1 * xr2,
                        xl2 * xr22, xr13, xr12 * xr2, xr1 * xr22, xr23,
                        xl14, xl13 * xl2, xl13 * xr1, xl13 * xr2, xl12 * xl22,
                        xl12 * xl2 * xr1, xl12 * xl2 * xr2, xl12 * xr12,
                        xl12 * xr1 * xr2, xl12 * xr22,
                        xl1 * xl23, xl1 * xl22 * xr1, xl1 * xl22 * xr2,
                        xl1 * xl2 * xr12, xl1 * xl2 * xr1 * xr2,
                        xl1 * xl2 * xr22, xl1 * xr13, xl1 * xr12 * xr2,
                        xl1 * xr1 * xr22, xl1 * xr23,
                        xl24, xl23 * xr1, xl23 * xr2, xl22 * xr12,
                        xl22 * xr1 * xr2,
                        xl22 * xr22, xl2 * xr13, xl2 * xr12 * xr2,
                        xl2 * xr1 * xr22, xl2 * xr23,
                        xr14, xr13 * xr2, xr12 * xr22, xr1 * xr23, xr24])

    return mat


def direct_identification(xc1_identified,
                          xc2_identified,
                          direct_a,
                          direct_polynomial_form=3):
  """Identification of the points detected on both cameras left and right
  into the global 3D-space

  Args:
     xc1_identified : numpy.ndarray
         Points identified on the left camera
     xc2_identified : numpy.ndarray
         Points identified on the right camera
     direct_a : numpy.ndarray
         Constants of direct polynomial
     direct_polynomial_form : int, optional
         Polynomial form


  Returns:
     x_solution : numpy.ndarray
         Identification in the 3D space of the detected points
  """
  # Solve by direct method
  xl1, xl2 = xc1_identified[:, 0], xc1_identified[:, 1]
  xr1, xr2 = xc2_identified[:, 0], xc2_identified[:, 1]
  xl = xl1, xl2
  xr = xr1, xr2

  mat = DirectPolynom({'polynomial_form': direct_polynomial_form}).pol_form(xl,
                                                                            xr)
  x_solution = np.matmul(direct_a, mat)
  return x_solution


class Tridical(crappy.blocks.Block):

  def __init__(self,
               direct_a_file: str,
               trans_matrix_file: str,
               label: str) -> None:

    crappy.blocks.Block.__init__(self)
    self.direct_A = np.load(direct_a_file)
    self.direct_polynom_degree = 4
    self.M = np.load(trans_matrix_file)
    self.label = label
    self.cmd_labels = ['tl(s)', 'tr(s)', 'pix_l', 'pix_r']
    self.values = {}
    self.l0x = 0
    self.stock_x = []
    self.stock_y = []
    self.stock_z = []
    self.stock_exx = []
    self.stock_t = []
    self.Lid = []
    self.nb_loop = 0

  def prepare(self):

    for label in self.cmd_labels:
      self.values[label] = None

  def loop(self):

    data = self.recv_all_data()
    for label in self.cmd_labels:
      if label in data.keys():
        self.values[label] = data[label]
    if self.values['pix_l'] is None or self.values['pix_r'] is None:
      return

    all_pix = get_all_pix(np.array(self.values['pix_l'][0])[:, 0],
                          np.array(self.values['pix_l'][0])[:, 1],
                          np.array(self.values['pix_r'][0])[:, 0],
                          np.array(self.values['pix_r'][0])[:, 1])

    if self.nb_loop == 0:
      r_pix_trans = r_to_l_trans(all_pix[1], self.M)
      l_false = []
      for j in range(len(r_pix_trans)):
        print('ecart x')
        print(all_pix[0][j][0] - r_pix_trans[j][0])
        print('ecart y')
        print(all_pix[0][j][1] - r_pix_trans[j][1])
        if (all_pix[0][j][0] - r_pix_trans[j][0] > 160 or
                all_pix[0][j][0] - r_pix_trans[j][0] < 130 or 
                abs(all_pix[0][j][1] - r_pix_trans[j][1]) > 5):
          l_false.append([r_pix_trans[j], j])
      print(len(l_false))
      for j in range(len(l_false)):
        for k in range(len(l_false)):
          if 180 > all_pix[0][l_false[j][1]][0] - l_false[k][0][0] > 120:
            if abs(all_pix[0][l_false[j][1]][1] - l_false[k][0][1]) < 7:
              self.Lid.append([l_false[j][1], l_false[k][1]])
      print(len(self.Lid))
      self.nb_loop = 1

    right_buff = all_pix[1].copy()
    for j in range(len(self.Lid)):
      all_pix[1][self.Lid[j][0]] = right_buff[self.Lid[j][1]]

    out = {}
    print(all_pix)
    left, right = all_pix
    x_direct_solution = direct_identification(left, right, self.direct_A,
                                              self.direct_polynom_degree)
    x, y, z = x_direct_solution
    self.stock_x.append(y)
    self.stock_y.append(x)
    self.stock_z.append(z)  # ATTENTION ON A CHANGE X ET Y MINCE MINCE MINCE
    print(x)
    print(y)
    print(z)
    self.l0x = max(self.stock_x[0]) - min(self.stock_x[0])
    #    print(f'l0x: {self.l0x}')
    exx = (max(y) - min(y)) / self.l0x - 1
    print(f'exx: {exx}')
    out[self.label] = float(exx)
    out['t(s)'] = float(self.values['tr(s)'][0])
    self.stock_exx.append(float(exx))
    self.stock_t.append(float(self.values['tr(s)'][0]))
    if len(self.stock_exx) > 1:
      out['dexx'] = ((self.stock_exx[-1] - self.stock_exx[-2]) / (
          self.stock_t[-1] - self.stock_t[-2]))
    else:
      out['dexx'] = 0
    print(f'dexx: {out["dexx"]}')
    out['ztrid'] = z[-1]
    self.send(out)
    for label in self.cmd_labels:
      self.values[label] = None
