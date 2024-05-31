import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.animation as animation 
import scipy.constants as ct
from matplotlib.animation import FuncAnimation

#For the QM part, we require a simple 1D FDTD scheme




### Electric field ###
class ElectricField:
    def __init__(self, field_type, dt, Nt, omega = 1, amplitude=1.0):
        self.field_type = field_type
        self.amplitude = amplitude
        self.dt = dt
        self.omega = omega
        self.Nt = Nt
        self.t0 = self.Nt*self.dt/5
        self.sigma = self.t0/5

    def generate(self, t, **kwargs):
        if self.field_type == 'gaussian':
            return self._gaussian(t)
        elif self.field_type == 'sinusoidal':
            return self._sinusoidal(t, **kwargs)
        else:
            raise ValueError(f"Unknown field type: {self.field_type}")

    def _gaussian(self, t):
        return self.amplitude * np.exp(-0.5 * ((t - self.t0) / self.sigma) ** 2)

    def _sinusoidal(self, t):
        #add damping function
        return self.amplitude * np.cos(self.omega * t)*2/np.pi* np.arctan(t/self.t0)
    
### Vector Potential for velocity gauge ###
    
class vecpot:
    def __init__(self, dt, Nt, omega , amplitude=1.0):
     
        self.amplitude = amplitude
        self.dt = dt
        self.omega = omega
        self.Nt = Nt
        self.t0 = self.Nt*self.dt/5

    def generate(self, t):
        return -self.amplitude/self.omega * np.sin(self.omega * t)*2/np.pi* np.arctan(t/self.t0)



### Potential ###
class Potential:
    def __init__(self, m, omega,Ny,dy):
        self.m = m
        self.omega = omega
        self.Ny = Ny
        self.dy = dy
        
        
    #This will call the function depending on which type of source you have    
    def V(self):
        V = 0.5*self.m*self.omega**2* (np.linspace(-self.Ny//2*self.dy, self.Ny//2*self.dy,self.Ny))**2
        return V
    

#### QM ####
class QM:
    def __init__(self,order,Ny, Nt, dy, dt, hbar, m, q, alpha, potential, omega, N, gauge, omegafield = 1, amplitude = 1, field_type = None):
        self.Ny = Ny
        self.Nt = Nt
        self.dy = dy
        self.dt = dt
        self.hbar = hbar
        self.m = m
        self.q = q
        self.alpha=alpha
        self.result = None
        self.order = order
        self.potential = potential
        #self.efield = efield
        self.omega =omega
        self.N=N
        self.gauge = gauge
        self.field_type = field_type
        if gauge == 'length':
            if field_type ==  'gaussian':

                self.efield= ElectricField('gaussian',dt, Nt, amplitude = amplitude)

            elif field_type == 'sinusoidal':
                self.efield = ElectricField('sinusoidal',dt, Nt, omega = omegafield, amplitude = amplitude)
            else:
                raise ValueError(f"If length gauge is chosen, please provide either gaussian or sinusoidal as field type")
        if gauge == 'velocity':
            self.pot = vecpot(self.dt,self.Nt, omegafield , amplitude=amplitude)
        
        
       
        

        
        self.r = np.linspace(-self.Ny/2*self.dy, self.Ny/2*self.dy,self.Ny)
        
        self.PsiRe = (self.m*self.omega/(np.pi*self.hbar))**(1/4)*np.exp(-self.m*self.omega/(2*self.hbar)*(self.r-self.alpha*np.sqrt(2*self.hbar/(self.m*self.omega))*np.ones(self.Ny))**2)
        self.PsiIm = np.zeros(self.Ny)
        self.J = np.zeros(self.Ny)
        
       
        self.data_prob= []
        self.data_time = []
        self.data_mom=[]
        self.data_energy= []
        self.beamenergy = []
        self.datacurr = []
       

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
    
    def diff_for_energy(self,psi):
        
        psi= (-np.roll(psi,2) + 16*np.roll(psi,1) -30*psi + 16*np.roll(psi,-1)-np.roll(psi,-2))/(12*self.dy**2)
        psi[0] = 0
        psi[1]= 0
        psi[-1] = 0
        psi[-2]=0
        
        
        return psi

    ### Update ###
    def update(self,n):
        
        E = self.efield.generate((n)*self.dt)*np.ones(self.Ny)

        PsiReo = self.PsiRe
        self.PsiRe = PsiReo -self.hbar*self.dt/(2*self.m)*self.diff(self.PsiIm) - self.dt/self.hbar*(self.q*self.r*E-self.potential.V())*self.PsiIm
        
        
        self.PsiRe[0] = 0
        self.PsiRe[-1] = 0
        
        E = self.efield.generate((n+1/2)*self.dt)*np.ones(self.Ny)
        
        PsiImo = self.PsiIm
        self.PsiIm = PsiImo +self.hbar*self.dt/(2*self.m)*self.diff(self.PsiRe)+ self.dt/self.hbar*(self.q*self.r*E-self.potential.V())*self.PsiRe
        
        
        self.PsiIm[0] = 0
        self.PsiIm[-1] = 0
        
        #We need the PsiIm at half integer time steps -> interpol
        PsiImhalf = (PsiImo + self.PsiIm)/2
        self.J = self.hbar/(self.m*self.dy)*(self.PsiRe*np.roll(PsiImhalf,-1) - np.roll(self.PsiRe,-1)*PsiImhalf)
        self.J[0]=0
        self.J[-1]= 0
        self.J *= self.q*self.N
        

        Psi = self.PsiRe+ 1j*PsiImhalf
        momentum = np.conj(Psi)*-1j*self.hbar*1/(2*self.dy)*(np.roll(Psi,-1)-np.roll(Psi,1))
        momentum[0] = 0
        momentum[-1] = 0

        prob = self.PsiRe**2  + PsiImhalf**2
       
        energy = np.trapz((np.conj(Psi))*(-self.hbar**2/(2*self.m)*self.diff_for_energy(Psi)+self.potential.V()*(Psi)), dx=self.dy)
        beam_energy = np.trapz(np.conj(Psi)*(-self.q*self.r *E*(Psi)),dx = self.dy )

        self.data_time.append(n*self.dt)
        self.data_prob.append(prob)
        self.data_mom.append(np.trapz(momentum, dx= self.dy))
        self.data_energy.append(energy)
        self.beamenergy.append(beam_energy)
        self.datacurr.append(self.J)

    def update_vel(self, n):
       
        a = self.pot.generate(n*dt)

        
        PsiReo = self.PsiRe
        self.PsiRe = PsiReo -self.hbar*self.dt/(2*self.m)*self.diff(self.PsiIm) + self.dt/self.hbar*(self.potential.V())*self.PsiIm +self.dt*self.q/self.m*a/(2*self.dy)*(np.roll(self.PsiRe,-1)-np.roll(self.PsiRe,1))
        
        
        self.PsiRe[0] = 0
        self.PsiRe[-1] = 0

        a = self.pot.generate((n+1/2)*dt)
        
        PsiImo = self.PsiIm
        self.PsiIm = PsiImo +self.hbar*self.dt/(2*self.m)*self.diff(self.PsiRe) - self.dt/self.hbar*(self.potential.V())*self.PsiRe +self.dt*self.q/self.m*a/(self.dy*2)*(np.roll(self.PsiIm,-1)-np.roll(self.PsiIm,1))
        
        
        self.PsiIm[0] = 0
        self.PsiIm[-1] = 0

        #We need the PsiIm at half integer time steps -> interpol
        PsiImhalf = (PsiImo + self.PsiIm)/2
        self.J = self.hbar/(self.m*self.dy)*(self.PsiRe*np.roll(PsiImhalf,-1) - np.roll(self.PsiRe,-1)*PsiImhalf)
        self.J[0]=0
        self.J[-1]= 0


        Psi = self.PsiRe+ 1j*PsiImhalf
        momentum = np.conj(Psi)*(-1j*self.hbar*1/(2*self.dy)*(np.roll(Psi,-1)-np.roll(Psi,1)) - self.q*a*Psi)
        momentum[0] = 0
        momentum[-1] = 0

        prob = self.PsiRe**2  + PsiImhalf**2
        energy = np.trapz((self.PsiRe-1j*self.PsiIm)*(-self.hbar**2/(2*self.m)*self.diff(self.PsiRe+1j*self.PsiIm)+self.potential.V()*(self.PsiRe+1j*self.PsiIm))+ np.conj(Psi)*1j*self.hbar*q/(self.m)*a/(2*self.dy)*(np.roll(Psi,-1)-np.roll(Psi,1)), dx = self.dy)

        self.data_time.append(n*self.dt)
        self.data_prob.append(prob)
        self.data_mom.append(np.trapz(momentum, dx = self.dy))
        self.data_energy.append(energy)
    


    def calcwave(self):
        if self.gauge == 'length':
            for n in range (self.Nt):
                self.update(n)

        elif self.gauge == 'velocity':
            for n in range (self.Nt):
                self.update_vel(n)

    
    def expvalues(self,type):
        
        if type == 'position':
            exp = []
            for el in self.data_prob:
                exp.append(np.trapz(el*self.r*self.dy, dx = self.dy))
            return(exp)
                       
        if type == 'momentum':
            plt.plot(self.data_mom)
            plt.show()
            

        if type == 'energy':
            plt.plot(self.beamenergy)
            #plt.plot(self.data_energy)
            plt.show()

        if type == 'Continuity':
            self.lhs = []
            self.rhs = []
            self.explhs = []
            self.exprhs = []
            self.exptot = []
            
            for i in range(1,len(self.data_time)-1):
              
                    
        
                datacurrhalf = 1/2* (self.datacurr[i] + self.datacurr[i-1])
                val1 = -(datacurrhalf- np.roll(datacurrhalf,1))[1:]/dy

                self.lhs.append(val1)
                self.explhs.append(np.trapz(val1[1:-1],dx= self.dy))
                
       
                val2 = self.q*(self.data_prob[i] - self.data_prob[i-1])[1:]/dt
                self.rhs.append(val2)
                self.exprhs.append(np.trapz(val2[1:-1],dx= self.dy))

                self.exptot.append(np.trapz(val1[1:-1] -val2[1:-1],dx= self.dy))

                if i == self.Nt//3:
                    self.vallt = -val1
                    self.valrt = val2
                    

                

    def heatmap (self):
        probsel = self.data_prob[::100]
        plt.imshow(np.array(probsel).T)
        plt.xlabel(r'Normalised time $\frac{t}{\Delta t}$[]')
        plt.ylabel(r' Normalised Electron Position $\frac{y}{\Delta y}$ []')
        plt.title('Probability density of the electron as a function of time')
        plt.colorbar(label = r'$|\psi|^2$')
        plt.tight_layout()
        plt.show()


    def animate(self):
       
        self.data_probsel = self.data_prob[::50]
        self.data_timesel= self.data_time[::50]
        fig, ax = plt.subplots()

       
        line, = ax.plot([], [])

        def initanim():
            ax.set_xlim(0, len(self.data_probsel[0]))  
            ax.set_ylim(0, np.max(self.data_probsel))  
            return line,

        # Define the update function
        def updateanim(frame):
            line.set_data(np.arange(len(self.data_probsel[frame])), self.data_probsel[frame])
            return line,


      
        anim = FuncAnimation(fig, updateanim, frames=len(self.data_timesel), init_func=initanim, interval = 20)

    
        plt.show()

    def animate_two_curves(self):
        data1 = self.lhs[::100]
        data2 = self.rhs[::100]
        fig, ax = plt.subplots()
        ax.set_title('Animation for the LHS and RHS of the continuïty equation')
        ax.set_xlabel('y [m]')
        xdata, ydata = [], []
        ln, = plt.plot([], [], 'r-')
        ln2, = plt.plot([], [], 'b-')
        def init():
            ax.set_xlim(0, len(data1[0]))
            ax.set_ylim(np.min(data1), np.max(data1))
            return ln, ln2

        def update(frame):
            xdata=np.arange(len(data1[frame]))
            ydata=data1[frame]
            ln.set_data(xdata, ydata)
            ydata=data2[frame]
            ln2.set_data(xdata, ydata)
            return ln, ln2
        
        ani = FuncAnimation(fig, update, frames=len(data1), init_func=init, interval=30)
        plt.show()

##########################################################

class Plotting:
    def __init__(self, type):
        self.type = type

    def plot(self, QMscheme1, QMscheme2=None, type=None):
        if self.type == 'Expectation':
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(15, 3))
            fig.suptitle("Results for a {0} pulse in the {1} gauge".format(QMscheme1.field_type, QMscheme1.gauge))
            ax1.plot(QMscheme1.data_time[::50], QMscheme1.expvalues('position')[::50])
            ax1.set_title('Position')
            ax1.set_xlabel('t [s]')
            ax1.set_ylabel('y [m]')
            ymin, ymax = ax1.get_ylim()
            ax1.set_ylim(ymin * 1.5, ymax * 1.5)

            ax2.plot(QMscheme1.data_time[::50], QMscheme1.data_mom[::50])
            ymin, ymax = ax2.get_ylim()
            ax2.set_ylim(ymin * 1.5, ymax * 1.5)
            ax2.set_xlabel('t [s]')
            ax2.set_ylabel(r'$P_{kin}$ [$\frac{kg \cdot m}{s}$]')
            ax2.set_title('Kinetic Momentum')


            ax3.plot(QMscheme1.data_time[::50], QMscheme1.data_energy[::50])
            ax3.set_title("Kinetic + Potential Energy")
            ax3.set_xlabel('t [s]')
            ax3.set_ylabel('E [J]')
            plt.subplots_adjust(wspace=0.5)
            plt.tight_layout()
            plt.show()

        if self.type == 'Continuity': 
            QMscheme1.expvalues(self.type)
            plt.plot(QMscheme1.r[1:],QMscheme1.vallt, label = r'$\nabla \cdot J_q$')
            plt.plot(QMscheme1.r[1:],QMscheme1.valrt, label = r'$q\frac{\partial}{\partial_t}\rho_q$')
            plt.plot(QMscheme1.r[1:],QMscheme1.vallt+QMscheme1.valrt, label = r'$q\frac{\partial}{\partial_t}\rho_q + \nabla \cdot J_q$')
            
            plt.xlabel(r'Position y [m]')
            
            plt.ylabel(r'[$\frac{A}{m}$]')
            plt.title('Continuity equation')
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            QMscheme1.animate_two_curves()

        if self.type == 'Heatmap':
            QMscheme1.heatmap()

        if self.type == 'Compare':
            if type == 'gauges':
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(15, 5))
                fig.suptitle('Comparison between the Length an Velocity gauge')
                ax1.plot(QMscheme2.data_time[::50], QMscheme2.expvalues('position')[::50], label = 'Length')
                ax1.plot(QMscheme1.data_time[::50], QMscheme1.expvalues('position')[::50], label = 'Velocity')
                ax1.set_xlabel('t [s]')
                ax1.set_ylabel('y [m]')
                ymin, ymax = ax1.get_ylim()
                ax1.set_ylim(ymin * 1.5, ymax * 1.5)
                ax1.set_title('Position')

                ax2.plot(QMscheme2.data_time[::50], QMscheme2.data_mom[::50], label = 'Length')
                ax2.plot(QMscheme1.data_time[::50], QMscheme1.data_mom[::50], label = 'Velocity')
                ymin, ymax = ax2.get_ylim()
                ax2.set_ylim(ymin * 1.5, ymax * 1.5)
                ax2.set_title('Position')

                ax2.set_xlabel('t [s]')
                ax2.set_ylabel(r'$P_{kin}$ [$\frac{kg \cdot m}{s}$]')
                ax2.set_title('Kinetic Momentum')

                ax3.plot(QMscheme2.data_time[::50], QMscheme2.data_energy[::50], label = 'Length')
                ax3.plot(QMscheme1.data_time[::50], QMscheme1.data_energy[::50], label = 'Velocity')
                ax3.set_title("Kinetic + Potential Energy")
                ax3.set_xlabel('t [s]')
                ax3.set_ylabel('E [J]')

                ax1.legend()
                ax2.legend()
                ax3.legend()

                plt.subplots_adjust(wspace=0.5)
                plt.tight_layout()
                plt.show()
            elif type == 'order':
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(15, 5))
                fig.suptitle('Comparison between second and fourth order')
                ax1.plot(QMscheme1.data_time[::50], QMscheme1.expvalues('position')[::50], label = 'Second')
                ax1.plot(QMscheme2.data_time[::50], QMscheme2.expvalues('position')[::50], label  = 'Fourth')
                ax1.set_xlabel('t [s]')
                ax1.set_ylabel('y [m]')
                ymin, ymax = ax1.get_ylim()
                ax1.set_ylim(ymin * 1.5, ymax * 1.5)
                ax1.set_title('Position')

                ax2.plot(QMscheme1.data_time[::50], QMscheme1.data_mom[::50],label = 'Second')
                ax2.plot(QMscheme2.data_time[::50], QMscheme2.data_mom[::50], label  = 'Fourth')
                ymin, ymax = ax2.get_ylim()
                ax2.set_ylim(ymin * 1.5, ymax * 1.5)
                ax2.set_xlabel('t [s]')
                ax2.set_ylabel(r'$P_{kin}$ [$\frac{kg \cdot m}{s}$]')
                ax2.set_title('Kinetic Momentum')

                ax3.plot(QMscheme1.data_time[::50], QMscheme1.data_energy[::50],label = 'Second')
                ax3.plot(QMscheme2.data_time[::50], QMscheme2.data_energy[::50], label  = 'Fourth')
                ax3.set_title("Kinetic + Potential Energy")
                ax3.set_xlabel('t [s]')
                ax3.set_ylabel('E [J]')

                ax1.legend()
                ax2.legend()
                ax3.legend()

                plt.subplots_adjust(wspace=0.5)
                plt.tight_layout()
                plt.show()

eps0 = ct.epsilon_0
mu0 = ct.mu_0
hbar = ct.hbar #J⋅s
m = ct.electron_mass*0.15
q = -ct.elementary_charge 


dy = 0.25e-10

c = ct.speed_of_light # m/s
Sy = 1 # !Courant number, for stability this should be smaller than 1
dt = Sy*dy/c


Ny = 500
Nt = 40000
N = 1 #particles/m2


omegaHO = 50e14 #[rad/s]
alpha = 0

potential = Potential(m,omegaHO, Ny, dy)
potential.V()

gauge = 'velocity'
amplitude = 1e8
field_type = 'sinusoidal'
order = 'fourth'

QMscheme1 = QM(order,Ny, Nt, dy, dt, hbar, m, q, alpha, potential, omegaHO, N, gauge, omegafield = omegaHO, amplitude = amplitude, field_type = field_type)
QMscheme1.calcwave()


gauge = 'length'
amplitude = 1e8
field_type = 'sinusoidal'
order = 'fourth'

QMscheme2 = QM(order,Ny, Nt, dy, dt, hbar, m, q, alpha, potential, omegaHO, N, gauge, omegafield = omegaHO, amplitude = amplitude, field_type = field_type)
QMscheme2.calcwave()


plot = Plotting('Compare')
plot.plot(QMscheme1,QMscheme2,'gauges')

###########################################



gauge = 'length'
amplitude = 1e8
field_type = 'sinusoidal'
order = 'second'

QMscheme3 = QM(order,Ny, Nt, dy, dt, hbar, m, q, alpha, potential, omegaHO, N, gauge, omegafield = omegaHO, amplitude = amplitude, field_type = field_type)
QMscheme3.calcwave()

plot = Plotting('Compare')
plot.plot(QMscheme1,QMscheme2,'order')


# ################################################
gauge = 'length'
amplitude = 1e9
field_type = 'gaussian'
order = 'fourth'

QMscheme4 = QM(order,Ny, Nt, dy, dt, hbar, m, q, alpha, potential, omegaHO, N, gauge, omegafield = omegaHO, amplitude = amplitude, field_type = field_type)
QMscheme4.calcwave()


plot = Plotting('Expectation')
plot.plot(QMscheme4)

plot = Plotting('Continuity')
plot.plot(QMscheme4)

plot = Plotting('Heatmap')
plot.plot(QMscheme4)


