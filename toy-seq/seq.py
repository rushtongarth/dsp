import numpy as np

#c1,c2 = np.random.choice(6,p=[0,.2,.3,.2,.1,.2],size=2)
#p1,p2 = np.random.choice([1,4],size=2)
#print(c1,p1,c1,p2)

c1,p1,c2,p2 = 2,1,1,4
x0 = np.linspace(0,2*np.pi,100)
f  = lambda X:(np.sin(c1*X+p1)+2*np.sin(c2*X+p2))+(np.random.randn(*X.shape) * 0.1)
fx = f(x0)