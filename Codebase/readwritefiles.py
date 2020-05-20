####################################################################
#Read Matlab files into python using scipy
####################################################################
import scipy.io
def read_matdata(filename):
  data = scipy.io.loadmat(filename)
  return data
