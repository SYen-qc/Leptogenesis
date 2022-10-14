import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kn
from scipy.special import zeta
from scipy.integrate import odeint
from scipy import integrate
from mu_tau_class import mu_tau_NR
from odeintw import odeintw

def output_NR(t12,t23,t13,Delm21s,Delm31s,lambda_mag,theta,phi,hvev):

    output_NR = mu_tau_NR(t12,t23,t13,Delm21s,Delm31s,lambda_mag,theta,phi,hvev)
    output_NR.delta_slove()
    output_NR.active_neutrino()
    output_NR.Majorana_phase()
    output_NR.m_effective()
    output_NR.matrix()
    output_NR.MR_diagonalization()
    output_NR.epsilon()

    return output_NR.output

np.set_printoptions(precision=10)
output_array = output_NR(np.radians(33.44),np.radians(49.2),np.radians(8.57),7.42*1e-5,2.517*1e-3-0.5*7.42*1e-5,0.1,np.radians(60),np.radians(30),174)

g = 106.75
m_pl = 1.22*10**19
v = 246
#m = 0.05*10**(-9)
#m = (v**2*(output_array['Yukawa_MR_base'][0,0]*np.conj(output_array['Yukawa_MR_base'][0,0]))/(2*output_array['MR1'])).real
m = v**2*output_array['Yukawa_MR_base_pro'][0,0]/(2*output_array['MR1'])
h_t = 1 # 173*√2/246
#eps = 1e-6
eps = output_array['epsilon1']
#eps = output_array['epsilon1_test']
#print(m)
#print(eps)

# gamma/(M1^2T^3) decay N_1 to l,phi
def gamma1(z):
    return m/(8*np.pi**2*v**2)*z**2*kn(1,z)

# gamma/(M1^2T^3) scatter N_1,l to q_3,anti t
def gamma2(z):
    return 3*h_t**2*m/(64*np.pi**5*v**2)*z**2*integrate.quad(lambda x:(x**2-1)**2/x**2*kn(1,z*x),1,np.infty)[0]

# gamma/(M1^2T^3) scatter N_1,q_3 to l,t
def gamma3(z):
    return 3*h_t**2*m/(64*np.pi**5*v**2)*z**2*integrate.quad(lambda x:(x**2-1-np.log(x**2-1))*kn(1,z*x),1,np.infty)[0]

# N_1 Y^eq
def Y_eq(z):
    return 45/(2*np.pi**4*g)*z**2*kn(2,z)

# Boltzmann equation for N_1
def BEN(Y,z):
    dY_Ndz = -np.sqrt(45/(4*np.pi**3))*(45/np.pi**2)*(m_pl/g**1.5)*z*(Y/Y_eq(z)-1)*(gamma1(z))
    return dY_Ndz

# Boltzmann equation for B/3-L_α
def BEL(Y,z,c_l,c_phi):
    dY_Ndz = -np.sqrt(45/(4*np.pi**3))*(45/np.pi**2)*(m_pl/g**1.5)*z*(Y[0]/Y_eq(z)-1)*(gamma1(z))
    #y_N = Y[0]/Y_eq(z)
    dY_ldz = eps*dY_Ndz-np.sqrt(45/(4*np.pi**3*g))*np.pi**2*m_pl*z*(c_l/2+c_phi/2)*gamma1(z)*Y[1]
    return [dY_Ndz,dY_ldz]

def solve(c_l,c_phi):
    y0 = np.array([0+0j,0+0j], dtype=np.complex128)
    z = np.linspace(10**(-1),100,1000000)
    Y = odeintw(BEL,y0,z,args=(c_l,c_phi))
    Y = Y.T
    Y_N = Y[0]
    Y_l = Y[1]
    return z,Y_eq(z),Y_N,Y_l

if __name__ == '__main__':
    c_l = 1
    c_phi = 0
    #c_phi = 2/3
    #c_phi = 14/23
    #c_l = 78/115
    #c_phi = 56/115
    z,Y_N_eq,Y_N,Y_l = solve(c_l,c_phi)
    #print(Y_l)
    print(abs(np.pi**4*(43/11)*Y_l[-1]/(45*zeta(3))))

    plt.figure(figsize=(9.0, 7.0))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.07, 150)
    plt.ylim(1e-13, 2*1e-8)
    plt.plot(z,np.abs(np.pi**4*(43/11)*Y_l/(45*zeta(3))),label=r'$|\eta_{B}|$')
    plt.xlabel(r'$z=M_1/T$')
    plt.ylabel("$|\eta_{B}|$")
    #plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    #plt.savefig("test.pdf", bbox_inches="tight")
    #plt.savefig("/Users/shihyentseng/Documents/mu_tau_project/test.pdf")
    plt.show()
