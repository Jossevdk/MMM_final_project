import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation 
import copy
from scipy.sparse import csr_matrix
import time
import time
import psutil

from PMLX import PML_X
from PMLY2 import PML_Y


c0 = 299792458  # Speed of light in vacuum
eps0 = 8.854 * 10**(-12)
mu0 = 4 * np.pi * 10**(-7)

Z0 = np.sqrt(mu0 / eps0)

### Source ###
class Source:
    def __init__(self, x, y, J0, tc, sigma):
        self.x = x
        self.y = y
        self.J0 = J0
        self.tc = tc
        self.sigma = sigma

    def J(self, t):
        return self.J0 * np.exp(-(t - self.tc)**2.0 / (2 * self.sigma**2.0))

### UCHIE ###
class UCHIE:
    def __init__(self, Nx, Ny, dx, dy, dt, pml_kmax=None, pml_nl=None, m=None):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy
        self.dt = c0 * dt
        self.X = np.zeros((2 * Nx+2, Ny))
        self.Y = np.zeros((2 * Nx+2, Ny))

        self.pml_nl = pml_nl

        A1 = np.zeros((Nx+2, Nx+1))
        np.fill_diagonal(A1, 1)
        np.fill_diagonal(A1[1:], 1)

        A2 = np.zeros((Nx, Nx+1))
        np.fill_diagonal(A2, 1)
        np.fill_diagonal(A2[:,1:], 1)
        self.A2 = A2

        D1 = np.zeros((Nx, Nx+1))
        np.fill_diagonal(D1, -1/dx)
        np.fill_diagonal(D1[:,1:], 1/dx)

        D2 = np.zeros((Nx, Nx+1))
        np.fill_diagonal(D2, -1/dx)
        np.fill_diagonal(D2[:,1:], 1/dx)
        D2 = np.vstack((np.zeros(Nx+1), D2, np.zeros(Nx+1)))

    

        M_midden1 = np.hstack((A1 / self.dt, D2))
        M_midden2 = np.hstack((D1, A2 / self.dt))
        M = np.vstack((M_midden1, M_midden2))
        # A = pd.DataFrame(M)
        # A.columns = ['']*A.shape[1]
        # print(A.to_string(index=False))
        N_midden1 = np.hstack((A1 / self.dt, -D2))
        N_midden2 = np.hstack((-D1, A2 / self.dt))
        N = np.vstack((N_midden1, N_midden2))
        self.M_inv = np.linalg.inv(M)
        self.M_N = self.M_inv @ N
    
        self.ex0 = np.zeros((Nx+1, Ny + 1))
        if pml_kmax is not None and pml_nl is not None and m is not None:
            self.use_pml = False
            self.PMLR = PML_X(pml_nl + 2, Ny, dx, dy, dt, pml_kmax=pml_kmax, pml_nl=pml_nl, m=m)
            self.PMLL = PML_X(pml_nl + 2, Ny, dx, dy, dt, reverse= True,  pml_kmax=pml_kmax, pml_nl=pml_nl, m=m)
            self.PMLU = PML_Y(Nx + 2*pml_nl + 2, pml_nl+2, dx, dy, dt, pml_kmax=pml_kmax, pml_nl=pml_nl, m=m)
            self.PMLD = PML_Y(Nx + 2*pml_nl + 2, pml_nl+2, dx, dy, dt, reverse=True, pml_kmax=pml_kmax, pml_nl=pml_nl, m=m)
    def explicit(self):
        self.ex0[:, 1:-1] += self.dt / (self.dy) * (self.X[self.Nx+1:2 * self.Nx+2, 1:] - self.X[self.Nx+1:2 * self.Nx+2, :-1])

    def implicit(self, n, source):
        self.Y[self.Nx+2:2*self.Nx+2 , :] = self.A2@(self.ex0[:, 1:] - self.ex0[:, :-1])/self.dy
        self.Y[self.Nx+2 + int(source.x / self.dx), int(source.y / self.dy)] += -2 * (1 / Z0) * source.J(n * self.dt / c0)
        self.X = (self.M_N @ self.X + self.M_inv @ self.Y)
        #print(self.X[:,-1])
        
    def update(self, n, source):
        
        
        
        self.use_pml = True
        if self.use_pml or np.any(self.ex0[self.pml_nl+5, :] != 0):
            
            self.PMLL.X[4*self.PMLL.Nx+3, :] = self.X[self.Nx+2, :] #Hz CORRECT
            self.PMLL.X[self.PMLL.Nx, :] = self.X[1, :] #Ey CORRECT
            self.PMLL.ex0[-1, :] = self.ex0[1, :] #Ex CORRECT
            self.PMLL.update(n)
            self.X[self.Nx+1, :] = self.PMLL.X[4*self.PMLL.Nx+2, :] #Hz CORRECT
            self.X[0, :] = self.PMLL.X[self.PMLL.Nx-1, :] #Ey CORRECT
            self.ex0[0, :] = self.PMLL.ex0[-2, :] #Ex CORRECT

            

            self.PMLR.X[3*self.PMLR.Nx+3, :] = self.X[-2, :]
            self.PMLR.X[0, :] = self.X[-self.Nx-3, :]
            self.PMLR.ex0[0, :] = self.ex0[-2, :]
            self.PMLR.update(n)
            self.X[-1, :] = self.PMLR.X[3*self.PMLR.Nx+4, :]
            self.X[-self.Nx-2, :] = self.PMLR.X[1, :]
            self.ex0[-1, :] = self.PMLR.ex0[1, :]

            
            
            self.implicit(n, source)
            self.X[self.Nx+1:2*self.Nx+2, -1] =self.PMLU.X[1*self.PMLU.Nx+1+self.pml_nl+1:2*self.PMLU.Nx+2-self.pml_nl-1, 0]  #Hz CORRECT
            # self.PMLL.X[1*self.PMLL.Nx+1:2*self.PMLL.Nx+2, -1] = self.PMLU.X[1*self.PMLU.Nx+1:1*self.PMLU.Nx+1+self.pml_nl+3, 0]   #Hz CORRECT
            # self.PMLR.X[1*self.PMLR.Nx+1:2*self.PMLR.Nx+2, -1] =self.PMLU.X[2*self.PMLU.Nx+2-self.pml_nl-3:2*self.PMLU.Nx+2, 0]#Hz CORRECT
            self.X[self.Nx+1:2*self.Nx+2, 0] = self.PMLD.X[1*self.PMLU.Nx+1+self.pml_nl+1:2*self.PMLU.Nx+2-self.pml_nl-1, -1]  #Hz CORRECT

            self.X[0:self.Nx+1, -1] = self.PMLU.X[self.pml_nl+1:self.PMLU.Nx+1-self.pml_nl-1, 0]  #Ey CORRECT
            # self.PMLL.X[0:self.PMLL.Nx+1, -1] = self.PMLU.X[:self.pml_nl+3, 0] #Ey CORRECT
            # self.PMLR.X[0:self.PMLR.Nx+1, -1] = self.PMLU.X[2*self.PMLU.Nx+2-self.pml_nl-3:2*self.PMLU.Nx + 2, 0] #Ey CORRECT
            self.X[0:self.Nx+1, 0] = self.PMLD.X[self.pml_nl+1:self.PMLD.Nx+1-self.pml_nl-1, -1]
            
            self.explicit()
            
            self.ex0[:, -1] =self.PMLU.ex2[self.pml_nl+1:-self.pml_nl-1, 2] #Ex CORRECT
            # self.PMLL.ex0[:, -1] =self.PMLU.ex2[:self.pml_nl+3, 2] #Ex CORRECT
            # self.PMLR.ex0[:, -1]=self.PMLU.ex2[-self.pml_nl-3:, 2] #Ex CORRECT
            self.ex0[:, 0] =self.PMLD.ex2[self.pml_nl+1:-self.pml_nl-1, -3] #Ex CORRECT
            
            
            
            self.PMLU.ex2[self.pml_nl+1:-self.pml_nl-1, 1] = self.ex0[:, -2] #Ex CORRECT
            # self.PMLU.ex2[:self.pml_nl+3, 1] = self.PMLL.ex0[:, -2] #Ex CORRECT
            # self.PMLU.ex2[-self.pml_nl-3:, 1] = self.PMLR.ex0[:, -2] #Ex CORRECT
            self.PMLD.ex2[self.pml_nl+1:-self.pml_nl-1, -2] = self.ex0[:, 1] #Ex CORRECT
            
            
            self.PMLU.ex2[self.pml_nl+1:-self.pml_nl-1, 0] = self.ex0[:, -3] #Ex CORRECT
            # self.PMLU.ex2[:self.pml_nl+3, 0] = self.PMLL.ex0[:, -3] #Ex CORRECT
            # self.PMLU.ex2[-self.pml_nl-3:, 0] = self.PMLR.ex0[:, -3] #Ex CORRECT
            self.PMLD.ex2[self.pml_nl+1:-self.pml_nl-1, -1] = self.ex0[:, 2] #Ex CORRECT
            
            
            self.PMLU.X[1*self.PMLU.Nx+1+self.pml_nl+1:2*self.PMLU.Nx+2-self.pml_nl-1, 0] = self.X[self.Nx+1:2*self.Nx+2, -2]  #Hz CORRECT
            # self.PMLU.X[1*self.PMLU.Nx+1:1*self.PMLU.Nx+1+self.pml_nl+3, 0] = self.PMLL.X[1*self.PMLL.Nx+1:2*self.PMLL.Nx+2, -2]  #Hz CORRECT
            # self.PMLU.X[2*self.PMLU.Nx+2-self.pml_nl-3:2*self.PMLU.Nx+2, 0] =self.PMLR.X[1*self.PMLR.Nx+1:2*self.PMLR.Nx+2, -2] #Hz CORRECT
            self.PMLD.X[1*self.PMLU.Nx+1+self.pml_nl+1:2*self.PMLU.Nx+2-self.pml_nl-1, -1] = self.X[self.Nx+1:2*self.Nx+2, 1]  #Hz CORRECT
           
           
            self.PMLU.X[self.pml_nl+1:self.PMLU.Nx+1-self.pml_nl-1, 0] = self.X[0:self.Nx+1, -2]   #Ey CORRECT
            # self.PMLU.X[:self.pml_nl+3, 0] =self.PMLL.X[0:self.PMLL.Nx+1, -2] #Ey CORRECT
            # self.PMLU.X[2*self.PMLU.Nx+2-self.pml_nl-3:2*self.PMLU.Nx + 2, 0] =self.PMLR.X[0:self.PMLR.Nx+1, -2] #Ey CORRECT
            self.PMLD.X[self.pml_nl+1:self.PMLU.Nx+1-self.pml_nl-1, -1] = self.X[0:self.Nx+1, 1]   #Ey CORRECT
            
            self.PMLU.implicit(n)
            self.PMLU.explicit()
            
            
            
            
            
            
            
            
            
            
            
            
            #self.ex0[:, -1] = self.PMLU.ex1[:, 1] #Ex CORRECT
            

            # self.PMLD.ex2[self.pml_nl+1:-self.pml_nl-1, -1] = self.ex0[:, 1] #Ex CORRECT
            # self.PMLD.ex2[:self.pml_nl+3, -1] = self.PMLL.ex0[:, 1] #Ex CORRECT
            # self.PMLD.ex2[-self.pml_nl-3:, -1] = self.PMLR.ex0[:, 1] #Ex CORRECT
            # self.PMLD.explicit()
            # self.PMLD.implicit(n)
            
            # #self.ex0[:, -1] = self.PMLD.ex1[:, 1] #Ex CORRECT
            # self.X[self.Nx+1:2*self.Nx+2, 0] = self.PMLD.X[4*self.PMLD.Nx+4+self.pml_nl+1:-self.pml_nl-1, -1]  #Hz CORRECT
            # self.PMLL.X[3*self.PMLL.Nx+3:4*self.PMLL.Nx+4, 0] = self.PMLD.X[4*self.PMLD.Nx+4:4*self.PMLD.Nx+4+self.pml_nl+3, -1] #Hz CORRECT
            # self.PMLR.X[3*self.PMLR.Nx+3:4*self.PMLR.Nx+4, 0] = self.PMLD.X[-self.pml_nl-3:, -1] #Hz CORRECT

            # self.X[0:self.Nx+1, 0] = self.PMLD.X[self.pml_nl+1:self.PMLD.Nx+1-self.pml_nl-1, -1]  #Ey CORRECT
            # self.PMLL.X[0:self.PMLL.Nx+1, 0] = self.PMLD.X[:self.pml_nl+3, -1] #Ey CORRECT
            # self.PMLR.X[0:self.PMLR.Nx+1, 0] = self.PMLD.X[2*self.PMLD.Nx+2-self.pml_nl-3:2*self.PMLD.Nx + 2, -1] #Ey CORRECT
    
           
        
        
        return Z0 * self.X[:Nx - 1, :]

    def calculate(self, Nt, source):
        data_time = []
        data = []
        tracker =[[], []]
        for n in range(Nt):
            print(n)
            self.update(n, source)
            if n % 5 == 0:
                data_time.append(self.dt * n)
                data.append(Z0*self.ex0.T.copy())
                tracker[0].append((Z0*self.ex0[self.Nx//2,3*self.Ny//4]).copy())
                tracker[1].append((Z0*self.ex0[self.Nx//2,self.Ny//4]).copy())
            

        return data_time, data, tracker
    def animate_field(self, t, data):
        fig, ax = plt.subplots()

        ax.set_xlabel("x-axis [k]")
        ax.set_ylabel("y-axis [l]")
        # ax.set_xlim(0, Nx*dx)
        # ax.set_ylim(0, Ny*dy)

        label = "Field"
        
        # ax.plot(int(source.x/dx), int(source.y/dy), color="purple", marker= "o", label="Source") # plot the source

        cax = ax.imshow(data[3],vmin = -1e-13, vmax = 1e-13)
        ax.set_title("T = 0")

        def animate_frame(i):
            cax.set_array(data[i])
            ax.set_title("T = " + "{:.12f}".format(t[i]*1000) + "ms")
            return cax

        global anim
        
        anim = animation.FuncAnimation(fig, animate_frame, frames = (len(data)), interval=20)
        plt.show()

dx = 1e-10 # m
dy = 0.125e-9# ms

Sy = 0.8 # !Courant number, for stability this should be smaller than 1
dt = Sy*dy/c0
#print(dt)

Nx = 300
Ny = 300
Nt = 800

pml_nl = 20
pml_kmax = 20
m = 4

eps0 = 8.854 * 10**(-12)
mu0 = 4*np.pi * 10**(-7)
Z0 = np.sqrt(mu0/eps0)


xs = Nx*dx/2 
ys = Ny*dy/2

tc = dt*Nt/4
#print(tc)
sigma = tc/10
source = Source(xs, ys, 1, tc, sigma)

scheme = UCHIE(Nx, Ny, dx, dy, dt, pml_kmax=pml_kmax, pml_nl=pml_nl, m=m)
start_time = time.time()
data_time, data, tracker = scheme.calculate(Nt, source)

plt.plot(data_time, tracker[0])
plt.plot(data_time, tracker[1])
end_time = time.time()
print("Execution time:", end_time - start_time, "seconds")

# Optionally animate or plot data here
scheme.animate_field(data_time, data)
