import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation 
import scipy.constants as ct
from matplotlib.animation import FuncAnimation
import time
import psutil
#For the QM part, we require a simple 1D FDTD scheme

# from uchie_FAST import Source, UCHIE
# from PML_uchie import Source, UCHIE

#EM and QM classes containing all the necessary update function
import PML_uchie_update as EM
#import EM_update as EM
import QM_update as QM

eps0 = ct.epsilon_0
mu0 = ct.mu_0
hbar = ct.hbar #J⋅s
m = ct.electron_mass*0.15
q = -ct.elementary_charge
c0 = ct.speed_of_light 


Z0 = np.sqrt(mu0/eps0)




#### Coupling ####

class coupled:
    def __init__(self, EMsource, EMscheme, QMscheme, Nt):
        self.EMsource = EMsource
        self.EMscheme = EMscheme
        self.QMscheme= QMscheme
        self.Nt = Nt


    def calcwave(self):
        for n in range (self.Nt):
            # EMscheme.implicit(n, EMsource, QMscheme.J)
            # EMscheme.explicit()
            slice = int(1/2*(self.EMscheme.Ny-self.EMscheme.NyQM))
            self.QMscheme.update(self.EMscheme.X[self.EMscheme.QMxpos,slice:-slice],n)
            self.EMscheme.update(n,self.EMsource,self.QMscheme.J )
            #E = copy.deepcopy(Efield[2*Nx//4,:])
           
            #efield = EMscheme.X[:] #add the correct selection here
            



################################################
#all the input for EM part
dx = 0.25e-10 # m
dy = 0.25e-10# ms

Sy = 1 # !Courant number, for stability this should be smaller than 1
dt = Sy*dy/c0

Nx = 200
Ny =200
NyQM = int(3*Ny/4)
Nt = 1000

pml_nl = 10
pml_kmax = 4
eps0 = 8.854 * 10**(-12)
mu0 = 4*np.pi * 10**(-7)
Z0 = np.sqrt(mu0/eps0)


xs = Nx*dx/4 
ys = Ny*dy/2


J0 = 1
tc = dt*Nt/10
sigma = tc/5
QMxpos = Nx//2  #this is where the quantum sheet is positioned
mpml = 10
EMsource = EM.Source(xs, ys, J0, tc, sigma)
EMscheme = EM.UCHIE(Nx, Ny, NyQM, dx, dy, dt, QMxpos,pml_kmax = pml_kmax, pml_nl = pml_nl,m=mpml)


#############################################################
#QM input parameters


N = 10e7*dx #particles/m2
#NyQM = int(2*Ny/3)
order = 'fourth'
omega = 50e14 #[rad/s]
alpha = 0
potential = QM.Potential(m,omega, NyQM, dy)
#potential.V()

QMscheme = QM.QM(order,NyQM,dy, dt, hbar, m, q, alpha, potential, omega, N)

#############################################################
#start the coupled simulation
#Nt = 200

coupledscheme = coupled(EMsource,EMscheme, QMscheme, Nt)

coupledscheme.calcwave()
#coupledscheme.QMscheme.animate()
coupledscheme.QMscheme.expvalues('energy')
coupledscheme.EMscheme.animate_field()



# qm = coupled(order, Nx,Ny, dx, dy,dt, pml_kmax, pml_nl, J0, xs, ys, tc, sigma)

# start_time = time.time()


# res = qm.calc_wave( dy, dt, Ny, Nt,  hbar, m ,q ,potential, alpha, order,N)
# prob = res[3]
# probsel = prob[::100]


# process = psutil.Process()
# print("Memory usage:", process.memory_info().rss/(1024*1024), 'MB') # print memory usage
# print("CPU usage:", process.cpu_percent()) # print CPU usage

# end_time = time.time()


# print("Execution time: ", end_time - start_time, "seconds")

# qm.animate( dy, dt, Ny, Nt,  hbar, m ,q ,potential,alpha,order,N)
# qm.animate_field(res[0], res[5])



# qm.heatmap(dy, dt, Ny, Nt,  hbar, m ,q ,potential,alpha,order,N)
