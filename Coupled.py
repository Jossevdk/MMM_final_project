import os
os.environ["OMP_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["NUMEXPR_NUM_THREADS"] = "20"
NO_MKL = False
try:
    from sparse_dot_mkl import dot_product_mkl
except ImportError:
    print("No multithreaded dot product for sparse matrices...")
    NO_MKL = True
import numpy as np
from scipy.constants import mu_0 as mu0
from scipy.constants import epsilon_0 as eps0
import scipy.constants as ct
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation 
import matplotlib.patches as patch
from scipy.sparse import csr_matrix
import time
from tqdm import tqdm

### Parameters and universal constants ###
eps0 = ct.epsilon_0
mu0 = ct.mu_0
hbar = ct.hbar #J⋅s
m = ct.electron_mass*0.15
q = -ct.elementary_charge
c0 = ct.speed_of_light 
Z0 = np.sqrt(mu0/eps0)




### Create Gaussian magnetic current source ###
class Source:
    def __init__(self, x, y, J0, tc, sigma):
        self.x = x
        self.y = y
        self.J0 = J0
        self.tc = tc
        self.sigma = sigma
           
    def J(self, t):
        return self.J0*np.exp(-(t-self.tc)**2/(2*self.sigma**2))




### Functions needed for the PML in the YEE region ###
def genKy(Ny, Nx, pml_nl, m, dy):
    Ky = np.zeros((Nx+1, Ny+1))
    kmax = -np.log(np.exp(-16))*(m+1)/(2*np.sqrt(ct.mu_0/ct.epsilon_0)*pml_nl*dy)
    for iy in range(0, pml_nl):
        Ky[:,iy] = kmax*((pml_nl-1-iy)/pml_nl)**m
        Ky[:, -iy-1] = kmax*((pml_nl-1-iy)/pml_nl)**m
    return Ky

def genKx(Ny, Nx, pml_nl, m, dx):
    Kx = np.zeros((Nx+1, Ny+1))
    kmax = -np.log(np.exp(-16))*(m+1)/(2*np.sqrt(ct.mu_0/ct.epsilon_0)*pml_nl*dx)
    for ix in range(0, pml_nl):
        Kx[ix,:] = kmax*((pml_nl-1-ix)/pml_nl)**m
        Kx[-ix-1,:] = kmax*((pml_nl-1-ix)/pml_nl)**m
    return Kx



### CLass with the parameters for the UCHIE region ###
class QM_UCHIE_params:
    def __init__(self, Ly, n, N_sub, x_sub, QMxpos, QMscheme):
        self.Ly = Ly # Length of UCHIE region
        self.n = n # numbers of subgridding in one coarse grid
        self.N_sub = N_sub # Numbers of coarse grid to subgrid
        self.x_sub = x_sub # Location where the subgridding/UCHIE starts
        self.QMscheme = QMscheme # Location where the QM wire occurs in the UCHIE region
        self.QMxpos = QMxpos # Location where the QM wire occurs in the UCHIE region
        

        

### Creating the recorder on the locations (x, y) where the fields can be saved ###
class Recorder:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.data = []          # data of the field will be added in this list
        self.data_time = []     # data of the time will be added in this list


    # adding the measurement of the field to the data
    def save_data(self, field, t):
        self.data.append(field) # Save the field data here
        self.data_time.append(t) # Save the time data here
        



### Creating the UCHIE scheme with the QM wire ###
class QM_wire:
    def __init__(self, x1, x2, y1, y2, n, nx, ny, QMscheme, QMxpos, X, ex, M1_inv, M1_M2,ey, eymid, A_pol):
        self.x1 = x1 # Left interface of the UCHIE region on the x-axis
        self.x2 = x2 # Right interface of the UCHIE region on the x-axis
        self.y1 = y1 # Bottom interface of the UCHIE region on the y-axis
        self.y2 = y2 # Upper interface of the UCHIE region on the y-axis
        self.n = n # Numbers of subgridding in one coarse grid
        self.nx = nx # numbers of cells in x-direction
        self.ny = ny # Numbers of cells in y-direction
        self.QMscheme = QMscheme # Location where the QM wire occurs in the UCHIE region
        self.QMxpos = QMxpos # Location where the QM wire occurs in the UCHIE region
        self.X = X # Matrix with the unkown e_y and h_z discrete field quantities in UCHIE, see report
        self.ex = ex # Matrix with the unkown discrete field quantities in UCHIE
        self.M1_inv = M1_inv # Matrix inverse of LHS of the matrix equation, see report
        self.M1_M2 = M1_M2 # the product M1_inv @ M2
        self.ey = ey
        self.eymid = eymid # The e_y field on the location of the QM wi
        self.data = [] # Field data of the UCHIE region will be saved here
        self.A_pol = A_pol # Interpolation matrix needed for stitching, see report



###### The uniform YEE-UCHIE subgridding scheme ######
class Yee_UCHIE:
    def __init__(self, Nx, Ny, Nt, dx, dy, dt, sources, pml_nl, pml_m, qm_uchie_params = [], recorders=None, coupled=True):
        
        self.coupled = coupled # When True, coupling between QM and EM will hapen
        self.Nx = Nx # Numbers of coarse grids in x-direction
        self.Ny = Ny # Numbers of coarse grids in y-direction
        self.Nt = Nt # Number of update steps
        
        self.dx = dx # discretisation in coarse grid x-direction
        self.dy = dy # discretisation in coarse grid y-direction
        self.dt = dt # time discretisation 

        self.sources = sources
        self.recorders = recorders

        # Capital letters are used for the fields in the YEE
        self.Ex = np.zeros((Nx+1, Ny+1))
        self.Ey = np.zeros((Nx, Ny))
        self.Bz = np.zeros((Nx+1, Ny))
        self.Bzx = np.zeros((Nx+1, Ny))
        self.Bzy = np.zeros((Nx+1, Ny))


        # parameters needed for PML
        Ky = genKy(Ny, Nx, pml_nl, pml_m, dy)
        Kx = genKx(Ny, Nx, pml_nl, pml_m, dx)

        self.KxE = (2*np.full((Nx+1, Ny+1), ct.epsilon_0) - Kx*dt)/(2*np.full((Nx+1, Ny+1), ct.epsilon_0) + Kx*dt)
        self.KyE = (2*np.full((Nx+1, Ny+1), ct.epsilon_0) - Ky*dt)/(2*np.full((Nx+1, Ny+1), ct.epsilon_0) + Ky*dt)
        self.KxB = (2*np.full((Nx+1, Ny+1), ct.mu_0) - ct.mu_0*Kx*dt/ct.epsilon_0)/(2*np.full((Nx+1, Ny+1), ct.mu_0) + ct.mu_0*Kx*dt/ct.epsilon_0)
        self.KyB = (2*np.full((Nx+1, Ny+1), ct.mu_0) - ct.mu_0*Ky*dt/ct.epsilon_0)/(2*np.full((Nx+1, Ny+1), ct.mu_0) + ct.mu_0*Ky*dt/ct.epsilon_0)
        self.KxEB = (2*dt)/((2*np.full((Nx+1, Ny+1), ct.epsilon_0) + Kx*dt)*dx*ct.mu_0)
        self.KyEB = (2*dt)/((2*np.full((Nx+1, Ny+1), ct.epsilon_0) + Ky*dt)*dy*ct.mu_0)
        self.KxBE = (2*ct.mu_0*dt)/((2*np.full((Nx+1, Ny+1), ct.mu_0) +ct.mu_0*Kx*dt/ct.epsilon_0)*dx)
        self.KyBE = (2*ct.mu_0*dt)/((2*np.full((Nx+1, Ny+1), ct.mu_0) + ct.mu_0*Ky*dt/ct.epsilon_0)*dy)
        
        self.KxE = (self.KxE[:-1, :-1] +self.KxE[1:, :-1] +self.KxE[:-1, 1:] +self.KxE[1:, 1:])/4
        self.KxB = (self.KxB[1:-1, :-1] + self.KxB[1:-1, 1:])/2
        self.KyB = (self.KyB[1:-1, :-1] + self.KyB[1:-1, 1:])/2
        self.KxEB = (self.KxEB[:-1,:-1] +self.KxEB[1:,:-1]+self.KxEB[:-1,1:]+self.KxEB[1:,1:])/4
        self.KxBE = (self.KxBE[1:-1, :-1] + self.KxBE[1:-1, 1:])/2
        self.KyBE = (self.KyBE[1:-1, :-1] + self.KyBE[1:-1, 1:])/2
    

        # The field data will be saved here
        self.data_yee = []
        self.data_uchie = [[] for i in range(len(qm_uchie_params))]
        self.data_time = []

        # Putting all the QM and UCHIE subgridding region in one list
        self.QMwires = []
        for wire in qm_uchie_params:
            dx_f = dx/wire.n # discretisation step in x-direction in fine grid, UCHIE region

            nx = wire.n*wire.N_sub # Numbers of fine gridding x-direction in UCHIE region
            ny = int(wire.Ly/dy) # Numbers of grids in y-direction of the UCHIE region
            A_pol = (self.create_interpolation_matrix(wire.n, nx, wire.N_sub)) # Interpolation matrix needed for stitching
            
            X = np.zeros((2*nx+2, ny)) # Matrix with the unkown e_y and h_z discrete field quantities in UCHIE, see report
            ex = np.zeros((nx+1, ny+1)) # Matrix with the unkown discrete field quantities in UCHIE
            
            eymid = np.zeros((nx+1,ny )) # e_y field on the QM wire
            ey =  np.zeros((nx+1,ny ))
            #locations of the subgridding UCHIE region
            x1 = wire.x_sub
            x2 = x1 + wire.N_sub*dx
            y1 = (self.Ny-ny)//2 * dy
            y2 = y1 + ny * dy 

            # Differntiator matrix
            A_D = np.diag(-1 * np.ones(nx+1), 0) + np.diag(np.ones(nx), 1)
            A_D = A_D[:-1, :]

            #Interpolator matrix
            A_I = np.zeros((nx, nx + 1))
            np.fill_diagonal(A_I, 1)
            np.fill_diagonal(A_I[:,1:], 1)
            
            
            # Creating the matrix of LHS of the matrix equation (see report)
            M1_1 = np.zeros(2*nx+2) 
            M1_1[0] = 1/dx
            M1_1[nx+1] = 1/dt

            M1_2 = np.hstack((A_D/dx_f, A_I/dt))

            M1_3 = np.zeros(2*nx+2)
            M1_3[nx] = -1/dx
            M1_3[-1] = 1/dt

            M1_4 = np.hstack((eps0*A_I/dt, A_D/(mu0*dx_f)))

            M1 = np.vstack((M1_1, M1_2, M1_3, M1_4))

            M1_inv = (csr_matrix(np.linalg.inv(M1)))

            # Creating the matrix of the RHS of the matrix equation
            M2_1 = np.zeros(2*nx+2) 
            M2_1[0] = -1/dx
            M2_1[nx+1] = 1/dt
            M2_2 = np.hstack((-1/dx_f*A_D, 1/dt*A_I))
            M2_3 = np.zeros(2*nx+2)
            M2_3[nx] = 1/dx
            M2_3[-1] = 1/dt

            M2_4 = np.hstack((eps0/dt*A_I, -1/(mu0*dx_f)*A_D))

            M2 = np.vstack((M2_1, M2_2, M2_3, M2_4))
            M1_M2 = (csr_matrix(M1_inv @ M2))
            
            # discretisation of location to get index numbers
            x1 = int(round(x1/dx))
            x2 = int(round(x2/dx))
            y1 = int(round(y1/dy))
            y2 = int(round(y2/dy))

            self.QMwires.append(QM_wire(x1, x2, y1, y2, wire.n, nx, ny, wire.QMscheme, wire.QMxpos, X, ex, M1_inv, M1_M2, ey, eymid, A_pol))
        
        

    ## Calculations of the scheme, procedure from report is followed ##
    def calculate_fields(self):


        for time_step in tqdm(range(0, self.Nt)):


            # Update B field in Yee-region
            self.Bz_old = self.Bz 
            self.Bzy[1:-1, :] = self.KyB * self.Bzy[1:-1, :]  +  self.KyBE* (self.Ex[1:-1, 1:] - self.Ex[1:-1, :-1])  
            self.Bzx[1:-1, :] = self.KxB* self.Bzx[1:-1, :]  - self.KxBE* (self.Ey[1:, :] - self.Ey[:-1, :]) 

            # Add the magnetic source in the Yee region in the B-field
            for source in self.sources:
                self.Bzy[int(round(source.x/self.dx)), int(round(source.y/self.dy))] -= self.dt*source.J(time_step*self.dt)/2
                self.Bzx[int(round(source.x/self.dx)), int(round(source.y/self.dy))] -= self.dt*source.J(time_step*self.dt)/2

            self.stitching_B() # Stitching of the upper and lower interface to update B field in YEE region
            
            self.Bz = self.Bzx + self.Bzy


            # e_y fields will be used to update the QM scheme
            if self.coupled:
                for QMw in self.QMwires:
                    slice = int(1/2*(QMw.ny-QMw.QMscheme.Ny))
                    QMw.QMscheme.update(QMw.ey[QMw.QMxpos, slice:-slice],QMw.eymid[QMw.QMxpos, slice:-slice], time_step)
            

            # Field update in UCHIE region updated, bz and ey with implicit
            self.uchie_update()
            
            
            ### Update Ex and self.Ey in the Yee region ###
            self.Ex[1:-1, 1:-1] = self.KyE[1:-1, 1:-1]*self.Ex[1:-1, 1:-1]  +  self.KyEB[1:-1,1:-1] * (self.Bz[1:-1,1:] - self.Bz[1:-1,:-1])
            self.Ey = self.KxE *self.Ey  -  self.KxEB * (self.Bz[1:,:] - self.Bz[:-1,:])

            self.stitching_E() # Stitching of the left and right interface to update E_y field in YEE at the interface

            # Save the data's
            if time_step%(self.Nt/500)==0:
                self.data_yee.append(self.Bz.T/mu0)
                for data, QMw in zip(self.data_uchie, self.QMwires):
                    data.append(QMw.X[QMw.nx + 1:, :].T/mu0)
                
                self.data_time.append(time_step*self.dt)
            if self.recorders != None:
                for recorder in self.recorders:
                    recorder.save_data(self.Bz[int(round(recorder.x/self.dx)), int(round(recorder.y/self.dy))], time_step*self.dt)



    ## Create interpolation matrix for the stitching, see report ##
    def create_interpolation_matrix(self, n, nx, N_sub):
        A_pol = np.zeros((nx+1, N_sub+1))
        for i in range(N_sub):
            A_1 = np.arange(n+1)/n 
            A_2 = A_1[::-1]
            A = np.vstack((A_2, A_1)).T 
            A = A[:-1,:]
            A_pol[i*n:(i+1)*n, i:i+2] = A
        A_pol[-1, -1] = 1
        return A_pol
    


    ## Field update in UCHIE region updated, bz and ey with implicit and e_x explicit##
    def uchie_update(self):
        
        for QMw in self.QMwires:
            Y = (QMw.ex[:-1, 1:] + QMw.ex[1:, 1:] - QMw.ex[:-1, :-1] - QMw.ex[1:, :-1])/self.dy
            slice = int(1/2*(QMw.ny-QMw.QMscheme.Ny))
            if self.coupled:
                Y[QMw.QMxpos, slice :-slice]+= +2  * QMw.QMscheme.J # Adding the quantum current to the e_y field
            
            eyold = copy.deepcopy(QMw.X[:QMw.nx+1, :])
            
            U_left = 1/self.dy*(QMw.ex[0, 1: ] + self.Ex[QMw.x1-1, QMw.y1+1:QMw.y2+1] - QMw.ex[0, :-1] - self.Ex[QMw.x1-1, QMw.y1:QMw.y2])  +  1/self.dx*(self.Ey[QMw.x1-1, QMw.y1:QMw.y2] + self.Ey[QMw.x1-2, QMw.y1:QMw.y2])  -  1/self.dt*(self.Bz[QMw.x1-1, QMw.y1:QMw.y2] - self.Bz_old[QMw.x1-1, QMw.y1:QMw.y2]) # UCHIE stitching left interface
            U_right = 1/self.dy*(QMw.ex[-1, 1: ] + self.Ex[QMw.x2+1, QMw.y1+1:QMw.y2+1] - QMw.ex[-1, :-1] - self.Ex[QMw.x2+1, QMw.y1:QMw.y2])  -  1/self.dx*(self.Ey[QMw.x2, QMw.y1:QMw.y2] + self.Ey[QMw.x2+1, QMw.y1:QMw.y2])  -  1/self.dt*(self.Bz[QMw.x2+1, QMw.y1:QMw.y2] - self.Bz_old[QMw.x2+1, QMw.y1:QMw.y2]) # UCHIE stitching right interface
            
            if NO_MKL:
                QMw.X = QMw.M1_M2 @ QMw.X + QMw.M1_inv @ np.vstack((U_left, Y, U_right, np.zeros((QMw.nx, QMw.ny)))) # Implicit update for e_y and b_z 
            else:
                QMw.X = dot_product_mkl(QMw.M1_M2, QMw.X) + dot_product_mkl(QMw.M1_inv, np.vstack((U_left, Y, U_right, np.zeros((QMw.nx, QMw.ny))))) # Implicit update for e_y and b_z
            QMw.ex[:, 1:-1] = QMw.ex[:, 1:-1]  +  self.dt/(mu0*eps0*self.dy) * (QMw.X[QMw.nx + 1:, 1:] - QMw.X[QMw.nx + 1:, :-1]) # Explicit update of e_x
            QMw.ex[:, -1] = QMw.ex[:, -1]  +  self.dt/(mu0*eps0*self.dy) * (QMw.A_pol @ self.Bz[QMw.x1:QMw.x2+1, QMw.y2] - QMw.X[QMw.nx + 1:, -1]) # Stitching upper interface @ Uchie
            QMw.ex[:, 0] = QMw.ex[:, 0]  -  self.dt/(mu0*eps0*self.dy) * (QMw.A_pol @ self.Bz[QMw.x1:QMw.x2+1, QMw.y1-1] - QMw.X[QMw.nx + 1:, 0]) # Stitching down interface @ Uchie

            QMw.ey = QMw.X[:QMw.nx+1, :]
            QMw.eymid = 1/2*(eyold+QMw.X[:QMw.nx+1, :])
        
    
    ## Stitching conditions, updating the B fields in YEE at the upper and bottom interface ##
    def stitching_B(self):
        for QMw in self.QMwires:
            self.Bzx[QMw.x1: QMw.x2+1, QMw.y1:QMw.y2] = 0 # Set the B fields in the UCHIE region to zero, in order not the double count in the updates
            self.Bzx[QMw.x1:QMw.x2+1, QMw.y2] = self.Bzx[QMw.x1:QMw.x2+1, QMw.y2]  -  self.dt/self.dy * QMw.ex[::QMw.n, -1]/2 # Stitching upper interface  
            self.Bzx[QMw.x1:QMw.x2+1, QMw.y1-1] = self.Bzx[QMw.x1:QMw.x2+1, QMw.y1-1]  +  self.dt/self.dy * QMw.ex[::QMw.n, 0]/2  # Stitching lower interface 

            self.Bzy[QMw.x1: QMw.x2+1, QMw.y1:QMw.y2] = 0 # Set the B fields in the UCHIE region to zero, in order not the double count in the updates
            self.Bzy[QMw.x1:QMw.x2+1, QMw.y2] = self.Bzy[QMw.x1:QMw.x2+1, QMw.y2]  -  self.dt/self.dy * QMw.ex[::QMw.n, -1]/2 # Stitching upper interface 
            self.Bzy[QMw.x1:QMw.x2+1, QMw.y1-1] = self.Bzy[QMw.x1:QMw.x2+1, QMw.y1-1]  +  self.dt/self.dy * QMw.ex[::QMw.n, 0]/2  # Stitching lower interface 
    


    ## Stitching conditions, updating the E_y fields in Yee the left and right interface ##
    def stitching_E(self):
        for QMw in self.QMwires:
            self.Ey[QMw.x1-1, QMw.y1:QMw.y2] = self.Ey[QMw.x1-1, QMw.y1:QMw.y2]  -  self.dt/(self.dx*mu0*eps0) * QMw.X[QMw.nx+1, :] # stiching left interface
            self.Ey[QMw.x2, QMw.y1:QMw.y2] = self.Ey[QMw.x2, QMw.y1:QMw.y2]  +  self.dt/(self.dx*mu0*eps0) * QMw.X[-1, :] # stiching right interface
            
            self.Ex[QMw.x1:QMw.x2+1, QMw.y1:QMw.y2+1] = 0 # Fields in the UCHIE region set zero to avoid double counting
            self.Ey[QMw.x1:QMw.x2, QMw.y1:QMw.y2] = 0 # Fields in the UCHIE region set zero to avoid double counting
            
            
    
    ## function to animate to simulation ##
    def animate_field(self, v):

        fig, ax = plt.subplots()

        ax.set_xlabel("x-axis [m]")
        ax.set_ylabel("y-axis [m]")
        ax.set_xlim(0, (self.Nx+1)*self.dx)
        ax.set_ylim(0, (self.Ny+1)*self.dy)

        v = v 
        
        # plot the sources
        for source in self.sources:
            xs = source.x
            ys = source.y
            ax.plot(xs, ys+0.5*self.dy, color="purple", marker= "o", label="Source")
        for recorder in self.recorders:
            xr = recorder.x
            yr = recorder.y
            ax.plot(xr, yr, color="green", marker= "o", label="Recorder")


        cax = ax.imshow(self.data_yee[0], vmin = -v, vmax = v, origin='lower', extent = [0, (self.Nx+1)*self.dx, self.dy/2, self.Ny*self.dy])
        ax.set_title("t = 0")
        scaxes = []
        for QMw, data in zip(self.QMwires, self.data_uchie):
            x1 = QMw.x1*self.dx
            x2 = QMw.x2*self.dx
            y1 = QMw.y1*self.dy
            y2 = QMw.y2*self.dy
            
            subgrid1 = [x1, x2, y1, y2]
            
            scaxes.append(ax.imshow(data[0], vmin=-v, vmax=v, origin='lower', extent=subgrid1))
            rect=patch.Rectangle((x1, y1),x2-x1, y2-y1, alpha = 0.05, facecolor="grey", edgecolor="black")
            ymin = (int(1/2*(self.Ny-QMw.ny))+int(1/2*(QMw.ny-QMw.QMscheme.Ny)))*self.dy
            ymax = (self.Ny- int(1/2*(self.Ny-QMw.ny))-int(1/2*(QMw.ny-QMw.QMscheme.Ny)))*self.dy
            ax.vlines(x1+QMw.QMxpos//QMw.n*self.dx, ymin=ymin, ymax = ymax, color='red', linewidth=1)

            ax.add_patch(rect)
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label("$H_z$ [A/m]")


        def animate_frame(i):
            cax.set_array(self.data_yee[i])
            for scax, data in zip(scaxes, self.data_uchie):
                scax.set_array(data[i])
            ax.set_title("t = " + str(self.data_time[i]))#"{:.12f}".format(t[i]*1000) + "ms")
            return cax

        global anim
        
        anim = animation.FuncAnimation(fig, animate_frame, frames = (len(self.data_yee)), interval=20)
        return anim
