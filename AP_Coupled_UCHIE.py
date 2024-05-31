import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation 
import scipy.constants as ct
from matplotlib.animation import FuncAnimation
import time
import psutil


#EM and QM classes containing all the necessary update function
import AP_UCHIE_update as EM

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
          
            slice = int(1/2*(self.EMscheme.Ny-self.EMscheme.NyQM))
            self.QMscheme.update(self.EMscheme.X[self.EMscheme.QMxpos,slice:-slice],self.EMscheme.Xmid[self.EMscheme.QMxpos,slice:-slice],n)
            self.EMscheme.update(n,self.EMsource,self.QMscheme.J )
           
            



################################################
#all the input for EM part
dx = 0.25e-10 # m
dy = 0.25e-10# m



Sy = 0.8 # !Courant number, for stability this should be smaller than 1
dt = Sy*dy/c0

Nx = 301
Ny =301
NyQM = int(3*Ny/4)
Nt = 800

pml_nl = 20
pml_kmax = 20
eps0 = 8.854 * 10**(-12)
mu0 = 4*np.pi * 10**(-7)
Z0 = np.sqrt(mu0/eps0)


xs = Nx*dx/4
ys = Ny*dy/2


J0 = 1
tc = dt*Nt/4
sigma = tc/10
QMxpos = Nx//2  #this is where the quantum sheet is positioned
mpml = 4
EMsource = EM.Source(xs, ys, J0, tc, sigma)
EMscheme = EM.UCHIE(Nx, Ny, NyQM, dx, dy, dt, QMxpos,pml_kmax = pml_kmax, pml_nl = pml_nl,m=mpml)


#############################################################
#QM input parameters


N = 10e7*dx #particles/m2

order = 'fourth'
omega = 50e14 #[rad/s]
alpha = 0
potential = QM.Potential(m,omega, NyQM, dy)


QMscheme = QM.QM(order,NyQM,dy, dt, hbar, m, q, alpha, potential, omega, N)

#############################################################
#start the coupled simulation


coupledscheme = coupled(EMsource,EMscheme, QMscheme, Nt)

coupledscheme.calcwave()

coupledscheme.QMscheme.expvalues('energy')
coupledscheme.EMscheme.animate_field()


