'''
Created on Sep 12, 2017

@author: dexter
'''
from scipy import linalg
import numpy as np

def constructWeightMatrixby2Basis(x_size, y_size, v1, v2):
  w=np.zeros((x_size,y_size, v1.shape[0])).astype('float64')
  #w[0,0] = np.random.randint(256,size=(1, 1, v1.shape[0]))
  for xi in np.arange(0,x_size):
    for yi in np.arange(0,y_size):
      w[xi,yi] = (v1)*float(xi) + (v2)*float(yi)
  return w

def initializeWeight2DVector( sampleData, weight_x, weight_y ):
    U,s,Vh = linalg.svd(sampleData)
    b1=Vh[0,:]
    b2=Vh[1,:]
    w=constructWeightMatrixby2Basis(weight_x,weight_y,b1,b2)
    return w