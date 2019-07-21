import numpy as np
import scipy.linalg as la

class BasicSSA(object):
  def __init__(self,time_series):
    self.ts = time_series
    
  def embed(self,window):
    self.w = window
    X = la.hankel(self.ts,np.zeros(self.w))
    k = self.ts.size+1-self.w
    self.X = X[:k,:].T
  def get_embedding(self,window):
    self.embed(window)
    return self.X
  def decompose(self,ngroups):
    d = np.linalg.matrix_rank(self.X)
    self.ngroups = ngroups
    S = self.X @ self.X.T
    U,s,V = la.svd(S)
    xv = []
    for i in range(d):
      ui = np.expand_dims(U[:,i],1)
      vi = 1/np.sqrt(s[i]) * (self.X.T @ ui)
      xv.append(np.sqrt(s[i]) * (ui @ vi.T))
    grps = np.arange(1,len(S),self.ngroups)
    grp_split = np.array_split(np.array(xv),grps)
    self.groups = np.array([sum(x) for x in grp_split])
  def get_decomp(self,ngroups):
    self.decompose(ngroups)
    return self.groups
  def diag_avg(self,matrix):
    K,L = max(matrix.shape),min(matrix.shape)
    self.ret = np.zeros(K+L-1)
    mirror = matrix[:,::-1]
    for e,k in enumerate(range(1-K,L)):
      x = np.diag(mirror,-k)
      self.ret[e] = np.sum(x)/len(x)
  def get_diag_avg(self,matrix):
    self.diag_avg(matrix)
    return self.ret
  def grouping(self):
    self.components = [self.get_diag_avg(m) for m in self.groups]
    #return components
  
  def __call__(self,**kargs):
    obj = BasicSSA(self.ts)
    obj.embed(kargs['window'])
    obj.decompose(kargs['ngroups'])
    obj.grouping()
    return obj

