import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation 
import scipy.constants as ct
from matplotlib.animation import FuncAnimation



### Electric field ###
class ElectricField:
    def __init__(self, field_type, dt, amplitude=1.0):
        self.field_type = field_type
        self.amplitude = amplitude
        self.dt = dt

    def generate(self, t, **kwargs):
        if self.field_type == 'gaussian':
            return self._gaussian(t, **kwargs)
        elif self.field_type == 'sinusoidal':
            return self._sinusoidal(t, **kwargs)
        
        #add a third case where these is coupling with the EM part
        else:
            raise ValueError(f"Unknown field type: {self.field_type}")

    def _gaussian(self, t, t0=0, sigma=1):
        t0 = 20000*self.dt
        sigma = t0/5
        return self.amplitude * np.exp(-0.5 * ((t - t0) / sigma) ** 2)

    def _sinusoidal(self, t, omega=1):
        t0= 1000*self.dt
        #add damping function
        return self.amplitude * np.sin(omega * t)*2/np.pi* np.arctan(t/t0)



### Potential ###
class Potential:
    def __init__(self, m, omega,Ny,dy):
        self.m = m
        self.omega = omega
        self.Ny = Ny
        self.dy = dy
   
    def V(self):
        V = 0.5*self.m*self.omega**2* (np.linspace(-self.Ny//2*self.dy, self.Ny//2*self.dy,self.Ny))**2
        return V
    

#### QM ####
class QM:
    def __init__(self,order,Ny,dy, dt, hbar, m, q, alpha, potential, omega, N):
        self.Ny = Ny
        self.dy = dy
        self.dt = dt
        self.hbar = hbar
        self.m = m
        self.q = q
        self.alpha=alpha
        self.result = None
        self.order = order
        self.potential = potential
        self.omega =omega
        self.N=N
        
       
     
    #    #coherent state at y=0 for electron
        
        self.r = np.linspace(-self.Ny/2*self.dy, self.Ny/2*self.dy,self.Ny)
        self.PsiRe = (self.m*self.omega/(np.pi*self.hbar))**(1/4)*np.exp(-self.m*self.omega/(2*self.hbar)*(self.r-self.alpha*np.sqrt(2*self.hbar/(self.m*self.omega))*np.ones(self.Ny))**2)
        self.PsiIm = np.zeros(self.Ny)
        self.Jmid = np.zeros(self.Ny)
        self.J = np.zeros(self.Ny)
        

        self.data_prob= []
        self.data_time = []
        self.data_mom=[]
        self.data_energy= []
        self.beam_energy=[]
        self.data_current = []
        self.data_position= []

    def diff(self,psi):
        if self.order == 'second':
            psi= (np.roll(psi,1) -2*psi + np.roll(psi,-1))/self.dy**2
            psi[0] = 0
            psi[-1] = 0
            return psi
        elif self.order == 'fourth':
            psi= (-np.roll(psi,2) + 16*np.roll(psi,1) -30*psi + 16*np.roll(psi,-1)-np.roll(psi,-2))/(12*self.dy**2)
            psi[0] = 0
            psi[1]= 0
            psi[-1] = 0
            psi[-2]=0
        else:
            raise ValueError(f"Order schould be 'second' or 'fourth'")
        
        return psi

    ### Update ###
    def update(self,efield,efieldmid,n):
     
 
        PsiReo = self.PsiRe
        self.PsiRe = PsiReo -self.hbar*self.dt/(2*self.m)*self.diff(self.PsiIm) - self.dt/self.hbar*(self.q*self.r*efieldmid-self.potential.V())*self.PsiIm
        
        self.PsiRe[0] = 0
        self.PsiRe[-1] = 0
       
        PsiImo = self.PsiIm
        self.PsiIm = PsiImo +self.hbar*self.dt/(2*self.m)*self.diff(self.PsiRe)+ self.dt/self.hbar*(self.q*self.r*efield-self.potential.V())*self.PsiRe
        
        
        self.PsiIm[0] = 0
        self.PsiIm[-1] = 0
        
        #We need the PsiIm at half integer time steps -> interpol
        PsiImhalf = (PsiImo + self.PsiIm)/2
        self.J = self.hbar/(self.m*self.dy)*(self.PsiRe*np.roll(PsiImhalf,-1) - np.roll(self.PsiRe,-1)*PsiImhalf)
        self.J[0]=0
        self.J[-1]= 0
        self.J = self.q*self.N*self.J

        

        Psi = self.PsiRe+ 1j*PsiImhalf
        momentum = np.conj(Psi)*-1j*self.hbar*1/(2*self.dy)*(np.roll(Psi,-1)-np.roll(Psi,1))
        momentum[0] = 0
        momentum[-1] = 0

        prob = self.PsiRe**2  + PsiImhalf**2

        energy = np.trapz((np.conj(Psi))*(-self.hbar**2/(2*self.m)*self.diff(Psi)+self.potential.V()*(Psi)),dx =self.dy)
        beam_energy = np.trapz(np.conj(Psi)*(-self.q*self.r *efield*(Psi)),dx = self.dy)
        
        self.data_time.append(n*self.dt)
        self.data_prob.append(prob.copy())
        self.data_mom.append(np.trapz(momentum, dx = self.dy))
        self.data_energy.append(energy)
        self.beam_energy.append(beam_energy)
        self.data_current.append(np.trapz(self.J,dx = self.dy))
        self.data_position.append(np.trapz(prob*self.r*self.dy,dx = self.dy))
    
    
    def expvalues(self,type):
   
        if type == 'position':
            exp = []
            for el in self.data_prob:
                exp.append(np.sum(el*self.r*self.dy))
            plt.plot(exp)
            plt.show()            
        if type == 'momentum':
            plt.plot(self.data_mom)
            plt.show()

        if type == 'all':

            plt.plot(self.data_time,self.data_energy)
            plt.title('Energy')
            plt.xlabel('time [s]')
            plt.ylabel('Energy [J]')
            plt.show()
            
            plt.plot(self.data_time,self.beam_energy)
            plt.title('beam energy')
            plt.xlabel('time [s]')
            plt.ylabel('Energy [J]')
            plt.show()
            plt.plot(self.data_time,self.data_current)
            plt.title('Quantum Current')
            plt.xlabel('time [s]')
            plt.ylabel('Current [1/s]')
            plt.show()
            plt.plot(self.data_time,self.data_position)
            plt.title('Position')
            plt.xlabel('time [s]')
            plt.ylabel('y [m]')
            plt.show()
        

    def heatmap(self):
        probsel = self.data_prob[::100]
        plt.figure(figsize=(60, 6))
        plt.imshow(np.array(probsel).T)
        
        x_ticks = np.arange(0, len(probsel), step=100)  # adjust the step value as needed
        x_labels = x_ticks * self.dt*100  # replace your_time_step with the actual time step
        plt.xticks(x_ticks, x_labels)
        plt.xlabel('time [s]')
        plt.ylabel('Electron Position [m]')
        plt.colorbar(label = r'$|\psi|^2$')
        plt.show()





    def animate(self):
        fig, ax = plt.subplots()

        # Create an empty plot object
        line, = ax.plot([], [])

        def initanim():
            ax.set_xlim(0, len(self.data_prob[0]))  
            ax.set_ylim(0, np.max(self.data_prob))  
            return line,

        # Define the update function
        def updateanim(frame):
            line.set_data(np.arange(len(self.data_prob[frame])), self.data_prob[frame])
            return line,


      
        anim = FuncAnimation(fig, updateanim, frames=len(self.data_time), init_func=initanim, interval = 10)
        plt.show()

####################################################
