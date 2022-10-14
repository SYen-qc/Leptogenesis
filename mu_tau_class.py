import math
import pprint
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy import optimize

class mu_tau_nu_osc():
    def __init__(self,t12,t23,t13,Delm21s,Delm31s):

        self.t12, self.t23, self.t13, self.Delm21s, self.Delm31s = t12, t23, t13, Delm21s, Delm31s
        self.output = {}
        #print("Initialization of the procedure")

    def delta_slove(self):
        def V(delta):
            return np.array([ [np.cos(self.t12)*np.cos(self.t13), np.sin(self.t12)*np.cos(self.t13), np.sin(self.t13)*np.exp(-delta*1j)],
                            [-np.sin(self.t12)*np.cos(self.t23) - np.cos(self.t12)*np.sin(self.t23)*np.sin(self.t13)*np.exp(delta*1j), np.cos(self.t12)*np.cos(self.t23) - np.sin(self.t12)*np.sin(self.t23)*np.sin(self.t13)*np.exp(delta*1j), np.sin(self.t23)*np.cos(self.t13)],
                            [np.sin(self.t12)*np.sin(self.t23) - np.cos(self.t12)*np.cos(self.t23)*np.sin(self.t13)*np.exp(delta*1j), -np.cos(self.t12)*np.sin(self.t23) - np.sin(self.t12)*np.cos(self.t23)*np.sin(self.t13)*np.exp(delta*1j), np.cos(self.t23)*np.cos(self.t13)] ], dtype=np.complex128)
        
        def R2(delta):
            return ((V(delta)[1][0]*V(delta)[2][2]+V(delta)[1][2]*V(delta)[2][0])*np.conj(V(delta)[0][1]))/((V(delta)[1][1]*V(delta)[2][2]+V(delta)[1][2]*V(delta)[2][1])*np.conj(V(delta)[0][0]))

        def R3(delta):
            return ((V(delta)[1][0]*V(delta)[2][1]+V(delta)[1][1]*V(delta)[2][0])*np.conj(V(delta)[0][2]))/((V(delta)[1][1]*V(delta)[2][2]+V(delta)[1][2]*V(delta)[2][1])*np.conj(V(delta)[0][0]))
        
        def R2s(delta):
            return R2(delta)*np.conj(R2(delta))

        def R3s(delta):
            return R3(delta)*np.conj(R3(delta))
        
        def func(delta):
            return self.Delm21s*R2s(delta).real*(1-R3s(delta).real)-(self.Delm31s+self.Delm21s/2)*R3s(delta).real*(1-R2s(delta).real)

        def delta_sol(self):
            return optimize.brentq(func, np.radians(180), np.radians(360))

        def delta_sol_lower(self):
            return optimize.brentq(func, 0, np.radians(180))

        self.delta_solution = delta_sol(self)
        self.delta_lower_solution = delta_sol_lower(self)
        self.V = V(delta_sol(self))
        self.R2 = R2(delta_sol(self))
        self.R3 = R3(delta_sol(self))
        self.output['delta'], self.output['delta_lower'], self.output['V'], self.output['R2'], self.output['R3'] =  delta_sol(self), delta_sol_lower(self), V(delta_sol(self)), R2(delta_sol(self)), R3(delta_sol(self))

        return self.delta_solution, self.delta_lower_solution
    
    def active_neutrino(self):
        m1 = np.abs(self.R2)*np.sqrt(self.Delm21s/(1-np.abs(self.R2)**2))
        m2 = m1/np.abs(self.R2)
        m3 = m1/np.abs(self.R3)
        m_sum = m1+m2+m3
        
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.m_sum = m_sum
        self.output['m1'], self.output['m2'], self.output['m3'], self.output['m_sum'] = m1, m2, m3, m_sum

        return self.m1, self.m2, self.m3, self.m_sum
    
    def Majorana_phase(self):
        def alpha2(self):
            return np.angle(self.R2)

        def alpha3(self):
            return np.angle(self.R3)
        
        self.alpha2 = alpha2(self)
        self.alpha3 = alpha3(self)
        self.output["alpha2"], self.output["alpha3"] = alpha2(self), alpha3(self)

        return self.alpha2, self.alpha3
    
    def m_effective(self):
        m_eff = np.sqrt(((np.cos(self.t12)**2*np.cos(self.t13)**2*self.m1)+(np.sin(self.t12)**2*np.cos(self.t13)**2*self.m2*np.exp(1j*self.alpha2))+(np.sin(self.t13)**2*self.m3*np.exp(1j*self.alpha3)*np.exp(-1j*2*self.delta_solution))) * ((np.cos(self.t12)**2*np.cos(self.t13)**2*self.m1)+(np.sin(self.t12)**2*np.cos(self.t13)**2*self.m2*np.exp(-1j*self.alpha2))+(np.sin(self.t13)**2*self.m3*np.exp(-1j*self.alpha3)*np.exp(1j*2*self.delta_solution))))

        self.m_eff = m_eff
        self.output['m_eff'] = m_eff

        return self.m_eff

    def show_outputall(self):
        return self.output

class mu_tau_NR(mu_tau_nu_osc):
    def __init__(self,t12,t23,t13,Delm21s,Delm31s,lambda_mag,theta,phi,hvev):
        super().__init__(t12,t23,t13,Delm21s,Delm31s)

        self.lambda_mag, self.theta, self.phi, self.hvev = lambda_mag, theta, phi, hvev
    
    def matrix(self):
        
        self.U = np.array([ [np.cos(self.t12)*np.cos(self.t13), np.sin(self.t12)*np.cos(self.t13)*np.exp(self.alpha2*1j/2.), np.sin(self.t13)*np.exp(self.alpha3*1j/2-self.delta_solution*1j)],
                          [-np.sin(self.t12)*np.cos(self.t23) - np.cos(self.t12)*np.sin(self.t23)*np.sin(self.t13)*np.exp(self.delta_solution*1j), (np.cos(self.t12)*np.cos(self.t23) - np.sin(self.t12)*np.sin(self.t23)*np.sin(self.t13)*np.exp(self.delta_solution*1j))*np.exp((self.alpha2*1j)/2.), np.sin(self.t23)*np.cos(self.t13)*np.exp((self.alpha3*1j)/2.)],
                          [np.sin(self.t12)*np.sin(self.t23) - np.cos(self.t12)*np.cos(self.t23)*np.sin(self.t13)*np.exp(self.delta_solution*1j), (-np.cos(self.t12)*np.sin(self.t23) - np.sin(self.t12)*np.cos(self.t23)*np.sin(self.t13)*np.exp(self.delta_solution*1j))*np.exp((self.alpha2*1j)/2.), np.cos(self.t23)*np.cos(self.t13)*np.exp((self.alpha3*1j)/2.)] ], dtype=np.complex128)
    
        self.lam_matrix = np.array([ [self.lambda_mag*np.cos(self.theta), 0, 0],
                                   [0, self.lambda_mag*np.sin(self.theta)*np.cos(self.phi), 0],
                                   [0, 0, self.lambda_mag*np.sin(self.theta)*np.sin(self.phi)] ])

        self.Dirac_mass = np.array([ [self.hvev*self.lambda_mag*np.cos(self.theta), 0, 0],
                                   [0, self.hvev*self.lambda_mag*np.sin(self.theta)*np.cos(self.phi), 0],
                                   [0, 0, self.hvev*self.lambda_mag*np.sin(self.theta)*np.sin(self.phi)] ])

        self.m = np.array([ [1e-9*self.m1, 0, 0],
                          [0, 1e-9*self.m2, 0],
                          [0, 0, 1e-9*self.m3] ])
        
        self.output['U'], self.output['lam_matrix'], self.output['Dirac_mass'], self.output['m'] = self.U, self.lam_matrix, self.Dirac_mass, self.m
        self.output['m_inv'], self.output['U_T'] = inv(self.m), self.U.T

        return self.U, self.lam_matrix, self.Dirac_mass, self.m

    def MR_diagonalization(self):
        MR = (-1) * self.Dirac_mass.T @ self.U @ inv(self.m) @ self.U.T @ self.Dirac_mass
        v = la.eigh(MR.conj().T @ MR)[1]
        MR_diag_temp = v.T @ MR @ v
        S = np.array([ [np.exp(np.angle(MR_diag_temp[0,0])*1j/2.), 0, 0],
                     [0, np.exp(np.angle(MR_diag_temp[1,1])*1j/2.), 0],
                     [0, 0, np.exp(np.angle(MR_diag_temp[2,2])*1j/2.)] ], dtype=np.complex128)
        Omega = v @ S.conj()

        #self.Yukawa_MR_base = Omega.T @ self.lam_matrix
        self.Yukawa_MR_base = self.lam_matrix @ Omega
        #self.Yukawa_MR_base = (self.lam_matrix @ Omega).conj()
        #self.Yukawa_MR_base_pro = self.Yukawa_MR_base @ self.Yukawa_MR_base.conj().T
        self.Yukawa_MR_base_pro = self.Yukawa_MR_base.conj().T @ self.Yukawa_MR_base
        #self.Yukawa_ulysses = self.Yukawa_MR_base.conj().T
        self.Yukawa_ulysses = self.Yukawa_MR_base.conj()
        #self.Yukawa_ulysses = self.Yukawa_MR_base
        self.Y_mag = abs(self.Yukawa_MR_base)
        self.Y_ang = np.angle(self.Yukawa_MR_base)
        
        self.MR_diag = (Omega.T @ MR @ Omega).real
        self.MR1 = (Omega.T @ MR @ Omega).real[0,0]
        self.MR2 = (Omega.T @ MR @ Omega).real[1,1]
        self.MR3 = (Omega.T @ MR @ Omega).real[2,2]
        self.output['MR1'], self.output['MR2'], self.output['MR3'] = self.MR1, self.MR2, self.MR3
        self.output['Yukawa_MR_base'], self.output['Yukawa_MR_base_pro'] = self.Yukawa_MR_base, self.Yukawa_MR_base_pro
        self.output['Y_mag'], self.output['Y_ang'] = abs(self.Yukawa_MR_base), np.angle(self.Yukawa_MR_base)
        self.output['Y_mag_ulysses'], self.output['Y_ang_ulysses'] = abs(self.Yukawa_ulysses), np.angle(self.Yukawa_ulysses)
        #self.output['Omega'], self.output['Omega_T'] = Omega, Omega.T

        return self.MR1, self.MR2, self.MR3, self.Yukawa_MR_base, self.Y_mag, self.Y_ang
    
    def epsilon(self):
        #self.epsilon1 = (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.square(self.Yukawa_MR_base_pro[0,1]).imag*g(np.square(self.MR2/self.MR1))+np.square(self.Yukawa_MR_base_pro[0,2]).imag*g(np.square(self.MR3/self.MR1)))
        self.epsilon1 = (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.square(self.Yukawa_MR_base_pro[1,0]).imag*g(np.square(self.MR2/self.MR1))+np.square(self.Yukawa_MR_base_pro[2,0]).imag*g(np.square(self.MR3/self.MR1)))

        #self.epsilon1_V_e = (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.conj(self.Yukawa_MR_base[0,0])*self.Yukawa_MR_base_pro[0,1]*self.Yukawa_MR_base[0,1]).imag*g(np.square(self.MR2/self.MR1)) + (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.conj(self.Yukawa_MR_base[0,0])*self.Yukawa_MR_base_pro[0,2]*self.Yukawa_MR_base[0,2]).imag*g(np.square(self.MR3/self.MR1))
        #self.epsilon1_S_e = (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.conj(self.Yukawa_MR_base[0,0])*self.Yukawa_MR_base_pro[1,0]*self.Yukawa_MR_base[0,1]).imag*(1/(1-np.square(self.MR2/self.MR1))) + (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.conj(self.Yukawa_MR_base[0,0])*self.Yukawa_MR_base_pro[2,0]*self.Yukawa_MR_base[0,2]).imag*(1/(1-np.square(self.MR3/self.MR1)))
        #self.epsilon1_V_m = (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.conj(self.Yukawa_MR_base[1,0])*self.Yukawa_MR_base_pro[0,1]*self.Yukawa_MR_base[1,1]).imag*g(np.square(self.MR2/self.MR1)) + (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.conj(self.Yukawa_MR_base[1,0])*self.Yukawa_MR_base_pro[0,2]*self.Yukawa_MR_base[1,2]).imag*g(np.square(self.MR3/self.MR1))
        #self.epsilon1_S_m = (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.conj(self.Yukawa_MR_base[1,0])*self.Yukawa_MR_base_pro[1,0]*self.Yukawa_MR_base[1,1]).imag*(1/(1-np.square(self.MR2/self.MR1))) + (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.conj(self.Yukawa_MR_base[1,0])*self.Yukawa_MR_base_pro[2,0]*self.Yukawa_MR_base[1,2]).imag*(1/(1-np.square(self.MR3/self.MR1)))
        #self.epsilon1_V_t = (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.conj(self.Yukawa_MR_base[2,0])*self.Yukawa_MR_base_pro[0,1]*self.Yukawa_MR_base[2,1]).imag*g(np.square(self.MR2/self.MR1)) + (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.conj(self.Yukawa_MR_base[2,0])*self.Yukawa_MR_base_pro[0,2]*self.Yukawa_MR_base[2,2]).imag*g(np.square(self.MR3/self.MR1))
        #self.epsilon1_S_t = (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.conj(self.Yukawa_MR_base[2,0])*self.Yukawa_MR_base_pro[1,0]*self.Yukawa_MR_base[2,1]).imag*(1/(1-np.square(self.MR2/self.MR1))) + (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.conj(self.Yukawa_MR_base[2,0])*self.Yukawa_MR_base_pro[2,0]*self.Yukawa_MR_base[2,2]).imag*(1/(1-np.square(self.MR3/self.MR1)))

        self.epsilon1_V_e = (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.conj(self.Yukawa_MR_base[0,1])*self.Yukawa_MR_base_pro[1,0]*self.Yukawa_MR_base[0,0]).imag*g(np.square(self.MR2/self.MR1)) + (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.conj(self.Yukawa_MR_base[0,2])*self.Yukawa_MR_base_pro[2,0]*self.Yukawa_MR_base[0,0]).imag*g(np.square(self.MR3/self.MR1))
        self.epsilon1_S_e = (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.conj(self.Yukawa_MR_base[0,1])*self.Yukawa_MR_base_pro[0,1]*self.Yukawa_MR_base[0,0]).imag*(1/(1-np.square(self.MR2/self.MR1))) + (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.conj(self.Yukawa_MR_base[0,2])*self.Yukawa_MR_base_pro[0,2]*self.Yukawa_MR_base[0,0]).imag*(1/(1-np.square(self.MR3/self.MR1)))
        self.epsilon1_V_m = (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.conj(self.Yukawa_MR_base[1,1])*self.Yukawa_MR_base_pro[1,0]*self.Yukawa_MR_base[1,0]).imag*g(np.square(self.MR2/self.MR1)) + (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.conj(self.Yukawa_MR_base[1,2])*self.Yukawa_MR_base_pro[2,0]*self.Yukawa_MR_base[1,0]).imag*g(np.square(self.MR3/self.MR1))
        self.epsilon1_S_m = (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.conj(self.Yukawa_MR_base[1,1])*self.Yukawa_MR_base_pro[0,1]*self.Yukawa_MR_base[1,0]).imag*(1/(1-np.square(self.MR2/self.MR1))) + (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.conj(self.Yukawa_MR_base[1,2])*self.Yukawa_MR_base_pro[0,2]*self.Yukawa_MR_base[1,0]).imag*(1/(1-np.square(self.MR3/self.MR1)))
        self.epsilon1_V_t = (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.conj(self.Yukawa_MR_base[2,1])*self.Yukawa_MR_base_pro[1,0]*self.Yukawa_MR_base[2,0]).imag*g(np.square(self.MR2/self.MR1)) + (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.conj(self.Yukawa_MR_base[2,2])*self.Yukawa_MR_base_pro[2,0]*self.Yukawa_MR_base[2,0]).imag*g(np.square(self.MR3/self.MR1))
        self.epsilon1_S_t = (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.conj(self.Yukawa_MR_base[2,1])*self.Yukawa_MR_base_pro[0,1]*self.Yukawa_MR_base[2,0]).imag*(1/(1-np.square(self.MR2/self.MR1))) + (1/(8*np.pi))*(1/(self.Yukawa_MR_base_pro[0,0]))*(np.conj(self.Yukawa_MR_base[2,2])*self.Yukawa_MR_base_pro[0,2]*self.Yukawa_MR_base[2,0]).imag*(1/(1-np.square(self.MR3/self.MR1)))

        self.epsilon1_test =  self.epsilon1_V_e + self.epsilon1_S_e + self.epsilon1_V_m + self.epsilon1_S_m + self.epsilon1_V_t + self.epsilon1_S_t
        self.output['epsilon1'], self.output['epsilon1_test'] = self.epsilon1,self.epsilon1_test


        return self.epsilon1

def g(x):
        return np.sqrt(x)*((1/(1-x))+1-(1+x)*np.log((1+x)/x))