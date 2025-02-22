'''
This file contains a tool kit for analyzing the performance of MPR spectrometers. 
It is main functionalities are:
    - evaluate efficiency of neutrons->proton conversion and selection
    - initializing an ensemble of protons
        + characteristic rays
        + Monte Carlo generation given a probability distribution for space/angles
        + Full synthetic neutron-proton conversion given a phase space distribution of neutrons
    - Transporting protons through the ion optics using transfer maps generated by COSY
    - Analysis of transported protons
        + Ion optical image analysis by phase portraits
        + x-y scattering in the focal plane
        + synthetic hodoscope binning
To build an MPR, one needs a foil, aperture, ion optics, and a hodoscope


Useful resources and data sources
ENDF Info -  https://www.oecd-nea.org/dbdata/data/endf102.htm#LinkTarget_12655
ENDF data -  https://www.nndc.bnl.gov/sigma/index.jsp?as=1&lib=endfb7.1&nsub=10

'''
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt 
from progress.bar import Bar


def FWHM(data, domain):
    '''
    given data defined over a domain, compute the full width at half max of data
    '''
    half_max = max(data) / 2.
    #find when function crosses line half_max (when sign of diff flips)
    #take the 'derivative' of signum(half_max - Y[])
    d = np.sign(half_max - (data[0:-1])) - np.sign(half_max - (data[1:]))
    #find the left and right most indexes
    left_idx = np.where(d > 0)[0]
    right_idx = np.where(d < 0)[-1]
    return domain[right_idx] - domain[left_idx] #return the difference (full width)

class acceptance:
    '''
    conversion foil upon which neutrons impinge and scatter protons and proton aperture which defines ion optical acceptance
    '''
    def __init__(self, rf, T, L, ra, srim_path, sigma_np_path, sigma_nC12_path, diffxs_path, Nrf=10000, Naf=10000, Nzf=10000, ap_type='circ'):
        '''
        rf - cm, foil radius
        T - um, foil thickness
        srim_path - string, path to srim data for material
        L - cm, distance from foil to aperture
        ra - cm, aperture radius
        sigma_np_path - path to total scattering cross section data
        diffxs_path - path to file containing differential cross section data (legendre coefficients)
        '''
        print('Initializing acceptance geometry')
        self.rho_m = 0.98 #g/cm3, mass density of CH2
        self.np = self.rho_m*6.022e23*1e6/7.01 #proton/m3, proton density of CH2
        self.nC = self.rho_m*6.022e23*1e6/14.02 #Carbon/m3, carbon density in CH2
        self.rf = rf/1e2 #m
        self.T = T/1e6 #m
        self.rf_grid=np.linspace(0, self.rf, Nrf) #cylindrical grid for sampling in foil
        self.af_grid=np.linspace(0, 2*np.pi, Naf)
        self.zf_grid=np.linspace(-self.T, 0, Nzf)
        self.SRIM_data = np.genfromtxt(srim_path, unpack = True)
        print('loaded SRIM data from', srim_path)
        self.L = L/1e2
        self.ra = ra/1e2
        self.sigmanp_data = np.genfromtxt(sigma_np_path, unpack=True, usecols=(0,1))
        print('loaded np elastic scattering cross sections from ', sigma_np_path)
        self.sigmanC12_data = np.genfromtxt(sigma_nC12_path, unpack=True, usecols=(0,1))
        print('loaded nC12 elastic scattering cross sections from ', sigma_nC12_path)
        self.diffxs_data = np.genfromtxt(diffxs_path, unpack=True)
        print('loaded differential scattering data from', diffxs_path)
        self.ap_type=ap_type
        print('')

    def get_foil_radius(self):
        '''
        returns radius in meters
        '''
        return self.rf

    def set_foil_radius(self, r, prints=False):
        self.rf = r/1e2
        if prints: print('Set conversion foil radius to %.2f cm' %r)

    def get_thickness(self):
        '''
        returns Thickness in meters
        '''
        return self.T
    
    def set_thickness(self, T, prints=False):
        self.T = T/1e6
        self.zf_grid=np.linspace(-self.T, 0, 100000)
        if prints: print('Set conversion foil thickness to %.1f um' %T)
    
    def get_separation(self):
        '''
        returns in meters. Note lack of setter for separation. foil-aperture separation impacts the transfer map and must match COSY inputs
        '''
        return self.L
    
    def set_separation(self, L):
        self.L = L/100

    def get_ap_radius(self):
        return self.ra
    
    def set_ap_radius(self, r, prints=False):
        self.ra = r/1e2
        if prints: print('Set proton aperture radius to %.2f cm' %r)

    def SP(self, E):
        '''
        calculate the stopping power, dE/dx [MeV/mm] of the foil for protons with energy E
        E - MeV, proton energy
        '''
        return np.interp(E, self.SRIM_data[0], self.SRIM_data[1]+self.SRIM_data[2])
    
    def SRIM(self, E0, L, N=1000):
        '''
        calculate the energy after slowing down in the foil (neglects straggling which for thin foils is small, this assumption gets worse as E decreases or T increases)
        E0 - MeV, initial energy
        L - m, distance traveled through material
        N - number of discretizations of path through foil
        '''
        dL = L/N
        E=E0
        for i in range(N):
            E-=self.SP(E)*dL*1e3 #SP is in MeV/mm, convert to MeV/m
        return E

    def sigma_np_total(self, E):
        '''
        return the cross section in m2 for elastic scattering
        E - MeV, incident particle energy
        '''
        return np.interp(E*1e6, self.sigmanp_data[0], self.sigmanp_data[1])*1e-28

    def sigma_nC12_total(self, E):
        '''
        return the cross section in m2 for elastic scattering
        E - MeV, incident particle energy
        '''
        return np.interp(E*1e6, self.sigmanC12_data[0], self.sigmanC12_data[1])*1e-28

    def diff_xs_CM(self, E, mu):
        '''
        compute the center of mass frame differential scattering cross section
        E in MeV
        mu is cos(theta_CM)
        '''
        l1=np.interp(E*1e6, self.diffxs_data[0], self.diffxs_data[1])
        l2=np.interp(E*1e6, self.diffxs_data[0], self.diffxs_data[2])
        l3=np.interp(E*1e6, self.diffxs_data[0], self.diffxs_data[3])
        l4=np.interp(E*1e6, self.diffxs_data[0], self.diffxs_data[4])
        l5=np.interp(E*1e6, self.diffxs_data[0], self.diffxs_data[5])
        l6=np.interp(E*1e6, self.diffxs_data[0], self.diffxs_data[6])
        #mu = np.cos(theta)
        return 0.5+1.5*l1*mu+2.5*l2*0.5*(3*mu**2-1)+3.5*l3*0.5*(5*mu**3-3*mu)+4.5*l4*0.125*(35*mu**4-30*mu**2+3)+5.5*l5*0.125*(63*mu**5-70*mu**3+15*mu)+6.5*l6*0.0625*(231*mu**6-315*mu**4+105*mu**2-5)
        
    def diff_xs_LAB(self, theta, E):
        '''
        theta is in the LAB frame
        theta - rad, scattering angle to evaluate (0, pi/2)
        E - incident neutron energy, MeV
        '''
        mu=1-2*np.cos(theta)**2
        return 4*np.cos(theta)*self.diff_xs_CM(E, mu)

    def generate_ray(self, E, Kinematic=False, SRIM=False, N_s=10000, z_samp='exp'):
        '''
        Generate a proton ray scattered by a neutron with normal incidence and enters the aperture. Scatter grid limited to improve calculation efficiency (dont make protons you know don't enter)
        E[MeV]: incident neutron energy
        Kinematic: bool, include cos^2 energy loss
        SRIM: bool, perform SRIM energy loss calc
        N_s: discretization of scatter angle
        z_samp: 'exp' or 'uni' to toggle sampling method in z
        returns ray variables x0, y0, theta_s, phi_s, E
        '''
        lim=np.arctan((self.rf+self.ra)/self.L) #max angle that is accepted in order to increase computational efficiency
        scatter_grid = np.linspace(0, lim, N_s) #don't sample beyond what could possibly be accepted
        dsdO = self.diff_xs_LAB(E, scatter_grid)
        dsdO /= np.sum(dsdO)
        acpt=False #bool tag if generated ray enters the aperture
        while acpt==False:
            rf = self.rf*np.sqrt(np.random.rand())
            thetaf = 2*np.pi*np.random.rand()
            x0=rf*np.cos(thetaf)
            y0=rf*np.sin(thetaf)
            if z_samp=='exp': Pz=np.exp(-(self.zf_grid+self.T)*(self.sigma_np_total(E)*self.np+self.sigma_nC12_total(E)*self.nC))
            if z_samp=='uni': Pz=np.ones_like(self.zf_grid)
            z0=np.random.choice(self.zf_grid, p=Pz/np.sum(Pz))
            #select random scattering angles using differential scattering information
            phi_s = 2*np.pi*np.random.rand() #azimuthal scatteirng angle
            theta_s = np.random.choice(scatter_grid, p=dsdO) #polar scattering angle
            acpt=self.check_ray(x0, y0, theta_s, phi_s) #bool, is proton in aperture?
            if acpt:
                #print('ray entered!', g)
                if Kinematic:
                    E=E*np.cos(theta_s)**2
                if SRIM:
                    L_SRIM = (-z0)/np.cos(theta_s)
                    E=self.SRIM(E, L_SRIM)
                x0+=z0*np.tan(theta_s)*np.cos(phi_s) #adjust initial x,y coords to account for transport through foil
                y0+=z0*np.tan(theta_s)*np.cos(phi_s)
            #else: print('generated ray not in aperture', g)
        return x0, y0, theta_s, phi_s, E

    def check_ray(self, x0, y0, theta_s, phi_s):
        '''
        returns true when a ray passes through an aperture, false otherwise
        '''
        xa=x0+self.L*np.tan(theta_s)*np.cos(phi_s)
        ya=y0+self.L*np.tan(theta_s)*np.sin(phi_s)
        if self.ap_type=='circ': 
            return xa**2+ya**2<=self.ra**2
        if self.ap_type=='rect': 
            return (np.abs(xa)<=self.ra)and(np.abs(ya)<=self.ra)
        else: print('Unsupported aperture type!')
    
    def get_efficiency(self, E, N=int(1e5), N_s=10000):
        '''
        estimate the intrinsic efficiency (p/n) of the spectrometer 
        calculate macroscopic cross section to get scattering efficiency
        rejection sample on scattering angle to get the fraction of scattered protons which actually enter the optics
        E - MeV, energy of incident neutron
        N - number of particles to simulate
        N_s - how many discritizations in scattering angle
        Return the intrinsic efficiency [protons accepted/neutrons incident]
        '''
        tot=N
        acpt=0
        #fraction of neutrons which undergo a scattering interaction on protons in the foil (assuming normal incidence), includes Carbon scatter competition
        eps_scat = self.np*self.sigma_np_total(E)*(1-np.exp(-(self.np*self.sigma_np_total(E)+self.nC*self.sigma_nC12_total(E))*self.T))/(self.np*self.sigma_np_total(E)+self.nC*self.sigma_nC12_total(E)) 
        #print('\n   CH2')
        #print(' np scattering efficiency = %.2e' %eps_scat)
        #print(' fraction of scattering reaction on hydrogen: ', self.np*self.sigma_np_total(E)/(self.np*self.sigma_np_total(E)+self.nC*self.sigma_nC12_total(E)))
        #print(' fraction of neutron flux which scatters: ', 1-np.exp(-(self.np*self.sigma_np_total(E)+self.nC*self.sigma_nC12_total(E))*self.T))
        #comparing pure H vs CH2
        #eps_scat = self.np*self.sigma_np_total(E)*(1-np.exp(-self.np*self.sigma_np_total(E)*self.T))/(self.np*self.sigma_np_total(E)) #fraction of neutrons which undergo a scattering interaction on protons in the foil (assuming normal incidence)
        #print('\n   pure H')
        #print(' np scattering efficiency = %.2e' %eps_scat)
        #print(' fraction of scattering reaction on hydrogen: ', self.np*self.sigma_np_total(E)/(self.np*self.sigma_np_total(E)))
        #print(' fraction of neutron flux which scatters: ', 1-np.exp(-(self.np*self.sigma_np_total(E))*self.T))
        scatter_grid = np.linspace(0, np.pi/2, N_s)
        dsdO = self.diff_xs_LAB(E, scatter_grid)
        dsdO /= np.sum(dsdO) #normalize scatter grid
        for n in range(tot):
            #select random points in the foil
            rf = self.rf*np.sqrt(np.random.rand()) #sqrt uniform random is proper sampling on a disk
            thetaf = 2*np.pi*np.random.rand()
            x0=rf*np.cos(thetaf)
            y0=rf*np.sin(thetaf)
            #select random scattering angles, using differential scattering cross section
            phi_s = 2*np.pi*np.random.rand() #azimuthal scatteirng angle
            theta_s = np.random.choice(scatter_grid, p=dsdO)
            xa=x0+self.L*np.tan(theta_s)*np.cos(phi_s)
            ya=y0+self.L*np.tan(theta_s)*np.sin(phi_s)
            #Check if the ray passes through the aperture
            if self.check_ray(x0, y0, theta_s, phi_s):
                acpt+=1
        eps_acpt = acpt/tot #fraction of scattered protons which enter the aperture
        return eps_acpt*eps_scat, eps_scat, eps_acpt
    
    def get_p_dist(self, En, N=int(1e2)):
        '''
        returns the proton energy distribution at exit of foil including SRIM and kinematic effects
        En - neutron energy [MeV]
        N - number of protons to simulate
        '''
        Eps = np.zeros(N)
        for i in range(N):
            r=self.generate_ray(En, True, True)
            Eps[i]=r[4]
        return Eps


class hodoscope:
    '''
    detector array at the focal plane. detectors are assumed to be centered on the final position of the reference ray
    '''
    def __init__(self, NL, NR, w, h):
        '''
        N_L - number of channels to the left (low energy)
        N_R - number of channels to the right (high energy)
        w - cm, detector width
        h - cm, detector height
        '''
        self.NL = NL
        self.NR = NR
        self.N = NL+NR+1 #plus 1 for central channel
        self.w = w/1e2
        self.h = h/1e2
        self.det_c = np.linspace(-(self.NL+0.5)*self.w, (self.NR+0.5)*self.w, self.N) #detector centers
    
    def get_detector_width(self):
        return self.w
    
    def set_detector_width(self, w):
        self.w = w/1e2
        self.det_c = np.arange(-(self.NL+0.5)*self.w, (self.NR+0.5)*self.w, self.N) #detector centers
    
    def get_detector_height(self):
        return self.h
    
    def set_detector_height(self, h):
        self.h =h/1e2

    def get_detector_centers(self):
        return self.det_c

def get_digits(num):
    '''
    get the digits of a number
    '''
    digits = np.zeros(6)
    i=0
    for d in str("{:.5f}".format(num/1e5)):
        if d != '.'and d!='e' and d!='-': 
            digits[i] = int(d)
            i+=1
    return digits

class MPR:
    '''
    This class represents a full MPR system
    '''
    def __init__(self, acceptance, map_path, refE, hodoscope):
        print('Initializing Magnetic Proton Recoil Spectrometer...')
        self.acceptance = acceptance
        self.map = np.genfromtxt(map_path, unpack=True)
        print('loaded COSY transfer map from ', map_path, '\n')
        self.refE=refE
        self.hodoscope=hodoscope
        #run horizontal axis initialization routine
        #
        #initialize proton ensembles
        self.beam_in = np.zeros(0)
        self.beam_out=np.zeros(0)
    
    def ChaRay(self, Nrf, Naf, Nra, Naa, NE, delE):
        '''
        initialize a set of characteristic rays to transport.
        delE in MeV, the max energy +/- from the reference energy
        characteristic rays are defined by selecting a polar grid of points in the foil to source from and a polar grid in the aperture to pass through
        A ray from each source gridpoint pass through each aperture grid point at each energy
        '''
        print('Initializing Proton ensemble to characteristic rays...')
        if Nrf ==0: self.beam_in = np.zeros((2*NE+1, 5))
        else:self.beam_in = np.zeros(((2*NE+1)*(Nrf*Naf+1)*(Nra*Naa+1), 5))
        if NE == 0: dE = delE
        else: dE = delE/(2*NE*self.refE)
        print('characteristic ray dE: ', dE*self.refE)
        i=0
        repeat=0
        for ne in range(2*NE+1):
            e = (NE-ne)*dE
            if Nrf == 0:
                self.beam_in[i] = [0, 0, 0, 0, e]
                i+=1
            else: 
                for nrf in range(Nrf+1):
                    for naf in range(Naf):
                        theta = 2*np.pi*naf/Naf
                        xx = self.acceptance.rf*np.cos(theta)*nrf/Nrf
                        yy =self.acceptance.rf*np.sin(theta)*nrf/Nrf
                        for nra in range(Nra+1):
                            for naa in range(Naa):
                                phi = 2*np.pi*naa/Naa
                                ax = np.arctan((xx+self.acceptance.ra*np.cos(phi)*nra/Nra)/self.acceptance.L)
                                ay = np.arctan((yy+self.acceptance.ra*np.sin(phi)*nra/Nra)/self.acceptance.L)
                                #self.beam_in[i]=[xx, -ax, yy, -ay, e]
                                r=False #repeat flag
                                for n in range(i):
                                    if np.all(self.beam_in[n]==[xx, -ax, yy, -ay, e]):
                                        repeat+=1
                                        r=True
                                if r == False:
                                    self.beam_in[i]=[xx, -ax, yy, -ay, e]
                                    i+=1
        print('Initialized', self.beam_in[:,0].size, 'protons')
        print('repeated ', repeat, ' times')

    def GenRays(self, E, f_E, Np, kinematics=False, SRIM=False, z_samp='exp', sup_print=False):
        '''
        Method for generating rays with arbitrary energy distribution
        E: Array of initial energies over which to sample [MeV]
        f_E: relative probabilities with which to sample (renormalized to have sum 1)
        NP: number of protons to simulate
        kinematics: boolean tag to include scattering in calculation of proton energy
        SRIM: boolean tag to include stopping power calculation to proton energy
        '''
        bar = Bar('generating initial proton trajectories...', max=Np)
        self.beam_in=np.zeros((Np,5)) #initialize input proton beam to correct size array
        n=0
        N=0
        #weight energy sampling distribution by the n,p scattering cross section
        f_E*=self.acceptance.sigma_np_total(E)
        f_E=f_E/np.sum(f_E)
        while n<Np-1:
            e=np.random.choice(E,p=f_E)
            x0, y0, th0, ph0, e = self.acceptance.generate_ray(e, kinematics, SRIM, z_samp=z_samp)
            if self.acceptance.check_ray(x0, y0, th0, ph0):
                xa=x0+self.acceptance.L*np.tan(th0)*np.cos(ph0)
                ax=np.arctan((xa-x0)/self.acceptance.L)
                ya=y0+self.acceptance.L*np.tan(th0)*np.sin(ph0)
                ay=np.arctan((ya-y0)/self.acceptance.L)
                de = (e-self.refE)/self.refE
                self.beam_in[n]=np.array([x0, ax, y0, ay, de])
                n+=1
                bar.next()
            N+=1
        bar.next()
        bar.finish()
        if not(sup_print): print('initialized %i protons from specified neutron distribution function \n' %self.beam_in.shape[0])

    def load_Map(self, Map_path):
        self.map = np.genfromtxt(Map_path, unpack=True)

    def Apply_Map(self, order = 1):
        '''
        Apply transfer map generated by COSY to the incident beam to generate the output beam
        '''
        bar = Bar('Applying order %i transfer map...' %order, max=len(self.beam_in[:,0]))
        #print('Applying order' , order, ' transfer map...')
        self.beam_out = np.zeros_like(self.beam_in)
        for i, b in enumerate(self.beam_in):
            self.beam_out[i]=[0,0,0,0,b[4]]
            for j, index in enumerate(self.map[-1]):
                digits = get_digits(index)
                if digits.sum()<=order:
                    xf = self.map[0, j]*b[0]**digits[0]*b[1]**digits[1]*b[2]**digits[2]*b[3]**digits[3]*b[4]**digits[5]
                    af = self.map[1, j]*b[0]**digits[0]*b[1]**digits[1]*b[2]**digits[2]*b[3]**digits[3]*b[4]**digits[5]
                    yf = self.map[2, j]*b[0]**digits[0]*b[1]**digits[1]*b[2]**digits[2]*b[3]**digits[3]*b[4]**digits[5]
                    bf = self.map[3, j]*b[0]**digits[0]*b[1]**digits[1]*b[2]**digits[2]*b[3]**digits[3]*b[4]**digits[5]
                    self.beam_out[i] += [xf, af, yf, bf, 0] 
            bar.next()
        bar.finish()
        print('Map Applied!\n')

    def assess_monoenergetic_performance(self, E, N=10000, order=5, kinematics = False, SRIM=False, drawfig=False, prints=False):
        '''
        check the performance for a monoenergetic incident neutron beam. 
        E: MeV - energy to assess performance at
        N: number of protons
        
        return average position, FWHM of position distribution, and resolution
        '''
        print('\nAssessing performance for', E, 'MeV monoenergetic neutrons...')
        self.GenRays(np.array([E]), np.array([1.]), N, kinematics, SRIM)
        self.Apply_Map(order)
        #get proton distribution parameters
        mean, std = norm.fit(self.beam_out[:,0])
        FWHM = 2.355*std
        R=self.map[0,5]/(2*std) #evaluate resolution from fwhm
        R_E=1/R
        #draw plots
        if drawfig:
            fig, ax = plt.subplots(1,2, figsize=(8,4))
            #scatter plot
            a=ax[1].scatter(self.beam_out[:,0], self.beam_out[:,2], c=self.beam_in[:,4]*self.refE+self.refE, s=0.7)
            fig.colorbar(a, ax = ax[1], label='proton energy[MeV]')
            ax[1].set_title('Proton scatter from %.1f MeV neutrons' %E)
            ax[1].set_xlabel('X [m]')
            ax[1].set_ylabel('Y [m]')
            ax[1].grid()
            #histogram
            ax[0].step(bins[:-1], x_hist)
            ax[0].set_title('Distribution of final X-locations')
            ax[0].set_xlabel('X [m]')
            ax[0].set_ylabel('frequency [arb]')
            ax[0].grid()
            plt.savefig('monoEperf')
        if prints:
            #print statements
            print('Ion Optical Image Parameters: ')
            print(' Mean position [cm]:  ', mean*100)
            print(' FWHM[cm]:    ', FWHM*100)
            print(' Resolution [%]', R_E*100)
            print('')
        return mean, 2*std, R_E


    def get_proton_density(self, dx=0.001, dy=0.001):
        '''
        This method calculates the density of proton impact sites in the focal plane
        dx, dy - resolution of coarse graining
        returns: proton density array
        '''
        xmax=np.max(self.beam_out[:,0])
        xmin=np.min(self.beam_out[:,0])
        X=np.linspace(xmin, xmax, int((xmax-xmin)/dx)+1)
        ymax=np.max(self.beam_out[:,2])
        ymin=np.min(self.beam_out[:,2])
        Y=np.linspace(ymin, ymax, int((ymax-ymin)/dy)+1)
        print(xmax, xmin, ymax, ymin)
        YY, XX = np.meshgrid(X, Y)
        P=np.zeros_like(XX) #Proton density
        for b in self.beam_out:
            #cycle through ensemble, convert coordinate to index corresponding to X-Y meshgrid covering beam area
            nx=int((b[0]-xmin)/dx)
            ny=int((b[2]-ymin)/dy)
            P[ny,nx]+=1/len(self.beam_out)
        return P, XX, YY


    def plot_output_XY(self, figname='output', draw_hodoscope=False ):
        fig, ax = plt.subplots(figsize=(4,4))
        ax.set_title('Proton Positions', fontsize=28)
        if draw_hodoscope:
            w=self.hodoscope.get_detector_width()
            h=self.hodoscope.get_detector_height()
            ax.vlines(self.hodoscope.det_c[0]-w/2, -h/2, h/2, colors='black', linewidth=0.5)
            for cent in self.hodoscope.det_c:
                ax.vlines(cent+w/2, -h/2, h/2, colors='black', linewidth=0.25)
            ax.hlines(np.array([h/2, -h/2]), self.hodoscope.det_c[0]-w/2, self.hodoscope.det_c[-1]+w/2, colors='black')
        else: ax.grid()
        a = ax.scatter(self.beam_out[:,0], self.beam_out[:,2], s=1.2, c=self.beam_in[:,4]*self.refE+self.refE, cmap='viridis')
        cbar=fig.colorbar(a, label = 'proton energy [MeV]')
        ax.legend()
        #ax.set_aspect(1)
        ax.set_xlabel('horizontal position [m]', fontsize=14)
        ax.set_ylabel('vertical position [m]', fontsize=14)   
        plt.savefig(figname)
    
    def plot_counts_v_position(self, Nbins=10):
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        ax.hist(self.beam_out[:, 0], bins=np.linspace(self.beam_out[:, 0].min(), self.beam_out[:, 0].max(), Nbins))
        plt.savefig('hist')



    def plotOutput(self):
        fig, ax = plt.subplots(2, 2, figsize=(5,5))
        fig.tight_layout(pad = 2.5)
        ax[0,0].scatter(self.beam_out[:, 0]*100, self.beam_out[:,2]*100,c=self.beam_in[:, 4]*self.refE+self.refE, s=0.8)
        ax[0,0].grid()
        ax[0,0].set_xlabel('X [cm]', labelpad = 1)
        ax[0,0].set_ylabel('Y [cm]', labelpad = 1)
        ax[0,0].set_title('xf-yf phase plot')
        
        ax[0,1].scatter(self.beam_out[:, 0]*100, self.beam_out[:, 1],c=self.beam_in[:, 4]*self.refE+self.refE, s=0.8)
        ax[0,1].grid()
        ax[0,1].set_xlabel('X [cm]', labelpad = 1)
        ax[0,1].set_ylabel('Theta_X [rad]', labelpad = 1)
        ax[0,1].set_title('xf-axf phase plot')
        
        ax[1,0].scatter(self.beam_out[:, 0]*100, self.beam_in[:, 4]*100, c=self.beam_in[:, 4]*self.refE+self.refE, s=0.8)
        ax[1,0].grid()
        ax[1,0].set_xlabel('X [cm]', labelpad = 1)
        ax[1,0].set_ylabel('dE/E [%]', labelpad = 1)
        ax[1,0].set_title('xf-Ei phase plot')
        
        ax[1,1].scatter(self.beam_out[:, 2]*100, self.beam_out[:, 3], c=self.beam_in[:, 4]*self.refE+self.refE, s=0.8)
        ax[1,1].grid()
        ax[1,1].set_xlabel('Y [cm]', labelpad = 1)
        ax[1,1].set_ylabel('Theta_Y [rad]', labelpad = 1)
        ax[1,1].set_title('yf-ayf phase plot')
        plt.savefig('output_plots')

    def DrawInputBeam(self):
        plt.figure(figsize=(4,6), dpi=600)
        plt.vlines(0, -self.acceptance.rf, self.acceptance.rf)
        plt.vlines(self.acceptance.L, -self.acceptance.ra, self.acceptance.ra)
        for b in self.beam_in:
            z=np.linspace(0, self.acceptance.L, 20)
            slope = np.tan(b[1])
            int = b[0]
            plt.plot(z, slope*z+int, alpha = 0.4)
        plt.savefig('incomingRays')
