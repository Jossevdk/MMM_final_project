import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation 
import copy
import pandas as pd
import scipy.sparse as sp
import time
import psutil


c0 = 299792458
eps0 = 8.854 * 10**(-12)
mu0 = 4*np.pi * 10**(-7)

Z0 = np.sqrt(mu0/eps0)



def lu_full_pivot(A):
    n = A.shape[0]
    P = np.eye(n)  # Permutation matrix for row swaps
    Q = np.eye(n)  # Permutation matrix for column swaps
    U = A.copy()
    L = np.eye(n)
    
    for k in range(n):
        # Find the pivot index
        max_index = np.unravel_index(np.argmax(np.abs(U[k:, k:])), U[k:, k:].shape)
        pivot_row = max_index[0] + k
        pivot_col = max_index[1] + k
        
        # Swap rows in U and P
        U[[k, pivot_row], k:] = U[[pivot_row, k], k:]
        P[[k, pivot_row], :] = P[[pivot_row, k], :]
        
        # Swap columns in U and Q
        U[:, [k, pivot_col]] = U[:, [pivot_col, k]]
        Q[:, [k, pivot_col]] = Q[:, [pivot_col, k]]
        
        # Swap rows in L to maintain lower triangular form, only for columns before k
        if k > 0:
            L[[k, pivot_row], :k] = L[[pivot_row, k], :k]
        
        # Compute multipliers and eliminate below
        for j in range(k+1, n):
            L[j, k] = U[j, k] / U[k, k]
            U[j, k:] -= L[j, k] * U[k, k:]
    
    return P, L, U, Q

### Source ###
class Source:
    def __init__(self, x, y, J0, tc, sigma):
        self.x = x
        self.y = y
        self.J0 = J0
        self.tc = tc
        self.sigma = sigma
        
    #This will call the function depending on which type of source you have    
    def J(self, t):
        #print(t)
        #return 10e7*np.cos(2*np.pi*2e7*t + 0.5)
        return self.J0*np.exp(-(t-self.tc)**2/(2*self.sigma**2))#*np.cos(10*t/self.tc)





#### UCHIE ####
class PML_X:
    def __init__(self,  Nx, Ny, dx, dy, dt,reverse = False, pml_kmax = None, pml_nl = None, m=None):
        
        self.Nx = Nx
        self.Ny = Ny

        self.dx = dx
        self.dy = dy
        self.dt = c0*dt
        self.X = np.zeros((4*Nx+4, Ny)) # the first Nx+1 rows are the Ey fields, and the others the Bz fields
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

        D2[0, 0] = 0
        D2[-1, -1] = 0
        D2[0, 1] = 0
        D2[-1, -2] = 0

    
        I_E = np.eye(Nx + 1)
        I_H = np.eye(Nx + 1)
        
        
        m = 10
        pml_kxmax = pml_kmax
        pml_sigmax_max = (m+1)/(150*np.pi*dx)
        
        pml_kx = np.array([1 + (pml_kxmax -1)*(i/pml_nl)**m for i in range(0, pml_nl)])
        pml_sigmax = np.array([pml_sigmax_max*(i/pml_nl)**m for i in range(0, pml_nl)])
        if reverse:
            pml_kx = pml_kx[::-1]
            pml_sigmax = pml_sigmax[::-1]
        
        k_tot_E = np.hstack((np.ones(Nx+1 - pml_nl), pml_kx))
        sigma_tot_E = np.hstack((np.zeros(Nx + 1 - pml_nl), pml_sigmax))
        k_tot_H = np.hstack((np.ones(Nx+1 - pml_nl), pml_kx))
        sigma_tot_H = np.hstack((np.zeros(Nx + 1 - pml_nl), pml_sigmax))
        #print(k_tot_E, sigma_tot_E, k_tot_H, sigma_tot_H)
        # pml_kymax = 4
        # pml_sigmay_max = (m+1)/(150*np.pi*dy)
        # pml_ky = np.array([1 + (pml_kymax -1)*(i/pml_nl)**m for i in range(0, pml_nl)])
        # pml_sigmay = np.array([pml_sigmay_max*(i/pml_nl)**m for i in range(0, pml_nl)])
        #print(np.diag(k_tot_H/self.dt+Z0*sigma_tot_H/2))
        M1 = np.hstack((A1/self.dt,                np.zeros((Nx+2, Nx+1)),                          np.zeros((Nx+2, Nx+1)),     D2                   ))
        M2 = np.hstack((np.zeros((Nx, Nx+1)),       A2/self.dt,                                            D1,                  np.zeros((Nx, Nx+1))   ))
        M3 = np.hstack((-I_E/self.dt,              np.zeros((Nx+1, Nx+1)),                       np.diag(k_tot_E/self.dt+Z0*sigma_tot_E/2),              np.zeros((Nx+1, Nx+1))))
        M4 = np.hstack((np.zeros((Nx + 1, Nx+1)),  -I_H/self.dt,                                  np.zeros((Nx + 1, Nx+1)), np.diag(k_tot_H/self.dt+Z0*sigma_tot_H/2)))
       
        N1 = np.hstack((A1/self.dt,                np.zeros((Nx+2, Nx+1)),                          np.zeros((Nx+2, Nx+1)),                         -D2                  ))
        N2 = np.hstack((np.zeros((Nx, Nx+1)),       A2/self.dt,                                            -D1,                                          np.zeros((Nx, Nx+1))   ))
        N3 = np.hstack((-I_E/self.dt,              np.zeros((Nx+1, Nx+1)),                       np.diag(k_tot_E/self.dt-Z0*sigma_tot_E/2),        np.zeros((Nx+1, Nx+1))))
        N4 = np.hstack((np.zeros((Nx + 1, Nx+1)),  -I_H/self.dt,                                  np.zeros((Nx + 1, Nx+1)),                      np.diag(k_tot_H/self.dt-Z0*sigma_tot_H/2)))
       
        M = np.vstack((M1, M2, M3, M4))
        
        self.M_inv = np.linalg.inv(M)
        self.N = np.vstack((N1, N2, N3, N4))
        self.M_N = self.M_inv@self.N

        #explicit part
        
        self.ex0 = np.zeros((Nx+1, Ny+1))

        self.Y =   np.vstack((np.zeros((self.Nx+2, self.Ny)),self.A2@(self.ex0[:, 1:] - self.ex0[:, :-1])/self.dy, np.zeros((self.Nx+1, self.Ny)), np.zeros((self.Nx+1, self.Ny))))
         

       

    def explicit(self):
        
       
        self.ex0[:,1:-1] = self.ex0[:,1:-1] + self.dt/(self.dy)*(self.X[3*self.Nx+3:4*self.Nx+4,1:] - self.X[3*self.Nx+3:4*self.Nx+4,:-1])
        
        
  

    def implicit(self, n):
        self.Y[self.Nx+2:2*self.Nx+2 , :] = self.A2@(self.ex0[:, 1:] - self.ex0[:, :-1])/self.dy
        #self.Y[self.Nx + int(source.x/self.dx), int(source.y/self.dy)] += -2*(1/Z0)*source.J(n*self.dt/c0)
        
        self.X = self.M_N@self.X + self.M_inv@(self.Y)
    def update(self,n):
        self.implicit(n)
        self.explicit()
        return 

    def calculate(self, Nt):
        data_time = []
        data = []

        for n in range(0, Nt):
            self.implicit(n)
            self.explicit()
            data_time.append(self.dt*n)
            #data.append(copy.deepcopy((Z0*self.ex0.T)))
            #data.append((Z0*self.ex0.T))
            data.append(copy.deepcopy((self.X[3*self.Nx-1:4*self.Nx,:].T)))
            
        
        return data_time, data
    def animate_field(self, t, data):
        fig, ax = plt.subplots()

        ax.set_xlabel("x-axis [k]")
        ax.set_ylabel("y-axis [l]")
        # ax.set_xlim(0, Nx*dx)
        # ax.set_ylim(0, Ny*dy)

        label = "Field"
        
        # ax.plot(int(source.x/dx), int(source.y/dy), color="purple", marker= "o", label="Source") # plot the source
        cax = ax.imshow(data[0],vmin = -1e-16, vmax = 1e-16)
        ax.set_title("T = 0")

        def animate_frame(i):
            cax.set_array(data[i])
            ax.set_title("T = " + "{:.12f}".format(t[i]*1000) + "ms")
            return cax

        global anim
        
        anim = animation.FuncAnimation(fig, animate_frame, frames = (len(data)), interval=20)
        plt.show()




# dx = 1e-10 # m
# dy = 0.125e-9# ms

# Sy = 0.8 # !Courant number, for stability this should be smaller than 1
# dt = Sy*dy/c0
# #print(dt)
# Nx = 200
# Ny = 200
# Nt = 400

# pml_nl = 20
# pml_kmax = 4
# eps0 = 8.854 * 10**(-12)
# mu0 = 4*np.pi * 10**(-7)
# Z0 = np.sqrt(mu0/eps0)


# xs = Nx*dx/2 
# ys = Ny*dy/2

# tc = dt*Nt/4
# #print(tc)
# sigma = tc/10
# source = Source(xs, ys, 1, tc, sigma)


# scheme = PML_X(Nx, Ny, dx, dy, dt, pml_kmax = pml_kmax, pml_nl = pml_nl)
# start_time = time.time()

# data_time, data = scheme.calculate(Nt)

# process = psutil.Process()
# print("Memory usage:", process.memory_info().rss) # print memory usage
# print("CPU usage:", process.cpu_percent()) # print CPU usage

# end_time = time.time()


# print("Execution time: ", end_time - start_time, "seconds")

# scheme.animate_field(data_time, data)
         