import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation 
import copy
import scipy.sparse as sp
import time
import psutil


c0 = 299792458
eps0 = 8.854 * 10**(-12)
mu0 = 4*np.pi * 10**(-7)

Z0 = np.sqrt(mu0/eps0)

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
class PML_Y:
    def __init__(self,  Nx, Ny, dx, dy, dt,reverse = False, pml_kmax = None, pml_nl = None, m=None):
        
        self.Nx = Nx
        self.Ny = Ny

        self.dx = dx
        self.dy = dy
        self.dt = c0*dt
        self.X = np.zeros((5*Nx+5, Ny)) # the first Nx+1 rows are the Ey fields, and the others the Bz fields
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
  
        
        
        m = m
        pml_kxmax = pml_kmax
        pml_sigmax_max = (m+1)/(150*np.pi*dx)

        pml_kx = np.array([1 + (pml_kxmax -1)*(i/pml_nl)**m for i in range(0, pml_nl)])
        pml_sigmax = np.array([pml_sigmax_max*(i/pml_nl)**m for i in range(0, pml_nl)])

        k_tot_x = np.hstack((pml_kx[::-1], np.ones(Nx+1 - 2*pml_nl), pml_kx))
        sigma_tot_x = np.hstack((pml_sigmax[::-1], np.zeros(Nx + 1 - 2*pml_nl), pml_sigmax))


        pml_kymax = pml_kmax
        pml_sigmay_max = (m+1)/(150*np.pi*dy)
        
        
        self.M_inv = []
        self.N = []
        self.M_N = []
        self.Betay_min = []
        self.Betay_plus_inv = []
        start = 0
        stop = Ny+1
        step = 1
        if reverse:
            start = Ny+1-1
            stop = -1
            step = -1

        for j in range(start, stop, step):
        
            
            if j < Ny-pml_nl:
                k_tot_y = np.ones(Nx+1)
                sigma_tot_y = np.zeros(Nx+1)
            else:
                
                k_tot_y = np.full((Nx+1,), 1 + (pml_kymax -1)*((j-Ny+pml_nl)/pml_nl)**m)
                sigma_tot_y = np.full((Nx+1,), pml_sigmay_max*((j-Ny+pml_nl)/pml_nl)**m )
     
                
            M1 = np.hstack((A1/self.dt,                np.zeros((Nx+2, Nx+1)),                          np.zeros((Nx+2, Nx+1)),     D2,                       np.zeros((Nx+2, Nx+1))))
            M2 = np.hstack((np.zeros((Nx, Nx+1)),      A2@np.diag(k_tot_x/self.dt+Z0*sigma_tot_x/2),  np.zeros((Nx, Nx+1)),     np.zeros((Nx, Nx+1)),     D1))
            M3 = np.hstack((-I_E/self.dt,              np.zeros((Nx+1, Nx+1)),                        I_E/self.dt,              np.zeros((Nx+1, Nx+1)),   np.zeros((Nx+1, Nx+1))))
            M4 = np.hstack((np.zeros((Nx + 1, Nx+1)),  -I_E/self.dt,                                  np.zeros((Nx + 1, Nx+1)), np.diag(k_tot_y/self.dt+Z0*sigma_tot_y/2),              np.zeros((Nx+1, Nx+1))))
            M5 = np.hstack((np.zeros((Nx + 1, Nx+1)),  np.zeros((Nx + 1, Nx+1)),                      -np.diag(k_tot_y/self.dt+Z0*sigma_tot_y/2),             np.zeros((Nx + 1, Nx+1)), np.diag(k_tot_x/self.dt+Z0*sigma_tot_x/2)))

            N1 = np.hstack((A1/self.dt,                np.zeros((Nx+2, Nx+1)),                          np.zeros((Nx+2, Nx+1)),    -D2,                       np.zeros((Nx+2, Nx+1))))
            N2 = np.hstack((np.zeros((Nx, Nx+1)),      A2@np.diag(k_tot_x/self.dt-Z0*sigma_tot_x/2),  np.zeros((Nx, Nx+1)),     np.zeros((Nx, Nx+1)),     -D1))
            N3 = np.hstack((-I_E/self.dt,              np.zeros((Nx+1, Nx+1)),                        I_E/self.dt,              np.zeros((Nx+1, Nx+1)),   np.zeros((Nx+1, Nx+1))))
            N4 = np.hstack((np.zeros((Nx + 1, Nx+1)),  -I_E/self.dt,                                  np.zeros((Nx + 1, Nx+1)), np.diag(k_tot_y/self.dt+Z0*sigma_tot_y/2),              np.zeros((Nx+1, Nx+1))))
            N5 = np.hstack((np.zeros((Nx + 1, Nx+1)),  np.zeros((Nx + 1, Nx+1)),                      -np.diag(k_tot_y/self.dt-Z0*sigma_tot_y/2),             np.zeros((Nx + 1, Nx+1)), np.diag(k_tot_x/self.dt-Z0*sigma_tot_x/2)))
            M = np.vstack((M1, M2, M3, M4, M5))

            self.M_inv.append(np.linalg.inv(M))
            self.N.append(np.vstack((N1, N2, N3, N4, N5)))
            self.M_N.append(self.M_inv[-1]@self.N[-1])

            self.Betay_min.append(np.diag(k_tot_y/self.dt-Z0*sigma_tot_y/2))
            self.Betay_plus_inv.append(np.linalg.inv(np.diag(k_tot_y/self.dt+Z0*sigma_tot_y/2)))
 
        #explicit part
        self.Betay_min = [self.Betay_min[0]] + self.Betay_min
        self.Betay_plus_inv = [self.Betay_plus_inv[0]] + self.Betay_plus_inv
        self.Betay_min = [np.average([self.Betay_min[i], self.Betay_min[i+1]], axis = 0) for i in range(len(self.Betay_min)-1)]
        self.Betay_plus_inv = [np.average([self.Betay_plus_inv[i], self.Betay_plus_inv[i+1]], axis = 0) for i in range(len(self.Betay_plus_inv)-1)]
        
        self.Betax_min = np.diag(k_tot_x/self.dt-Z0*sigma_tot_x/2)
        self.Betaz_min = np.eye(Nx+1)/self.dt
        self.Betax_plus = np.diag(k_tot_x/self.dt+Z0*sigma_tot_x/2)
        self.Betaz_plus_inv = np.linalg.inv(np.eye(Nx+1)/self.dt)

        self.ex2 = np.zeros((Nx+1, Ny+1))
        self.ex2old = np.zeros((Nx+1, Ny+1))
        self.ex1 = np.zeros((Nx+1, Ny+1))
        self.ex1old = np.zeros((Nx+1, Ny+1))
        self.ex0 = np.zeros((Nx+1, Ny+1))

        self.Y =   np.vstack((np.zeros((self.Nx+2, self.Ny)),self.A2@(self.ex0[:, 1:] - self.ex0[:, :-1])/self.dy, np.zeros((self.Nx+1, self.Ny)), np.zeros((self.Nx+1, self.Ny)), np.zeros((self.Nx+1, self.Ny))))
         
     

    def explicit(self):
        
        self.ex1old = self.ex1.copy()
        self.ex2old = self.ex2.copy()
        for j in range(0, self.Ny+1):
            if j >=1 and j <= self.Ny-1:
                self.ex2[:,j] = self.ex2[:,j] + self.dt/(self.dy)*(self.X[3*self.Nx+3:4*self.Nx+4,j] - self.X[3*self.Nx+3:4*self.Nx+4,j-1])
            
            self.ex1[:,j] = self.Betay_plus_inv[j]@(self.Betay_min[j]@self.ex1[:,j] + (self.ex2[:,j] - self.ex2old[:,j])/self.dt)
            self.ex0[:,j] = self.Betaz_plus_inv@(self.Betaz_min@self.ex0[:,j] + self.Betax_plus@self.ex1[:,j] - self.Betax_min@self.ex1old[:,j])
        
            #self.ex0[:, j] += self.dt / (self.dy) * (self.X[self.Nx+1:2 * self.Nx+2, j] - self.X[self.Nx+1:2 * self.Nx+2, j-1])

        # self.ex1old = self.ex1

        # ex1 = self.ex1[:,1:-1] + self.dt/(self.dy)*(self.X[2*self.Nx+2:3*self.Nx+3,1:] - self.X[2*self.Nx+2:3*self.Nx+3,:-1])
        # for j in range(1, self.Ny-1):
        #     self.ex0[:,j]  = self.Betay_plus_inv[j]@(self.Betay_min[j]@self.ex0[:,j] + (ex1[:,j] - self.ex1old[:,j])/self.dt)
        # self.ex1[:,1:-1] = ex1
        # print(self.ex0)
        
  

    def implicit(self, n):
        self.Y[self.Nx+2:2*self.Nx+2 , :] = self.A2@(self.ex0[:, 1:] - self.ex0[:, :-1])/self.dy
        
        #self.Y[self.Nx + 2+ int(source.x/self.dx), int(source.y/self.dy)] += -2*(1/Z0)*source.J(n*self.dt/c0)
        for j in range(0, self.Ny):
            self.X[:, j] = self.M_N[j]@self.X[:,  j] +self.M_inv[j]@(self.Y[:, j] )
        
    def update(self,n):
        self.explicit()
        self.implicit(n)
        
        return 

    def calculate(self, Nt):
        data_time = []
        data = []

        for n in range(0, Nt):
            self.implicit(n)
            self.explicit()
            data_time.append(self.dt*n)
            #data.append(copy.deepcopy((Z0*self.ex0.T)))
            #data.append((Z0*self.ex.T))
            data.append(copy.deepcopy((self.X[2*self.Nx+2:3*self.Nx+3,:].T)))
            
        
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
        
        anim = animation.FuncAnimation(fig, animate_frame, frames = (len(data)), interval=200)
        plt.show()




# dx = 0.125e-10 # m
# dy = 0.125e-10# ms

# Sy = 1 # !Courant number, for stability this should be smaller than 1
# dt = Sy*dy/c0
# #print(dt)
# Nx = 100
# Ny = 100
# Nt = 100

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


# scheme = PML_Y(Nx, Ny, dx, dy, dt, pml_kmax = pml_kmax, pml_nl = pml_nl)
# start_time = time.time()

# data_time, data = scheme.calculate(Nt)

# process = psutil.Process()
# print("Memory usage:", process.memory_info().rss) # print memory usage
# print("CPU usage:", process.cpu_percent()) # print CPU usage

# end_time = time.time()


# print("Execution time: ", end_time - start_time, "seconds")

# scheme.animate_field(data_time, data)
         