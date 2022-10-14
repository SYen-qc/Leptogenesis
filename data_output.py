import numpy as np
from mu_tau_class import mu_tau_NR
#from mu_tau_class import epsilon

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

#np.set_printoptions(precision=10, suppress=True)
np.set_printoptions(precision=10)
output_array = output_NR(np.radians(33.44),np.radians(49.2),np.radians(8.57),7.42*1e-5,2.517*1e-3-0.5*7.42*1e-5,0.1,np.radians(60),np.radians(30),174)
#output_array = output_NR(np.radians(33.44),np.radians(49.2),np.radians(8.57),7.42*1e-5,2.517*1e-3,0.1,np.radians(60),np.radians(30),174)
print(output_array)
#print(output_array['Yukawa_MR_base'])

'''
data output
'''
data_name = np.array(["Y11_mag", "Y12_mag", "Y13_mag", "Y21_mag", "Y22_mag", "Y23_mag", "Y31_mag", "Y32_mag", "Y33_mag", "Y11_phs", "Y12_phs", "Y13_phs", "Y21_phs", "Y22_phs", "Y23_phs", "Y31_phs", "Y32_phs", "Y33_phs", "M1", "M2", "M3"])
data_value = np.array([output_array['Y_mag_ulysses'][0,0], output_array['Y_mag_ulysses'][0,1], output_array['Y_mag_ulysses'][0,2], output_array['Y_mag_ulysses'][1,0], output_array['Y_mag_ulysses'][1,1], output_array['Y_mag_ulysses'][1,2], output_array['Y_mag_ulysses'][2,0], output_array['Y_mag_ulysses'][2,1], output_array['Y_mag_ulysses'][2,2], output_array['Y_ang_ulysses'][0,0], output_array['Y_ang_ulysses'][0,1], output_array['Y_ang_ulysses'][0,2], output_array['Y_ang_ulysses'][1,0], output_array['Y_ang_ulysses'][1,1], output_array['Y_ang_ulysses'][1,2], output_array['Y_ang_ulysses'][2,0], output_array['Y_ang_ulysses'][2,1], output_array['Y_ang_ulysses'][2,2], np.log10(output_array['MR1']), np.log10(output_array['MR2']), np.log10(output_array['MR3'])])
#data_value = np.array([output_array['Y_mag'][0,0], output_array['Y_mag'][0,1], output_array['Y_mag'][0,2], output_array['Y_mag'][1,0], output_array['Y_mag'][1,1], output_array['Y_mag'][1,2], output_array['Y_mag'][2,0], output_array['Y_mag'][2,1], output_array['Y_mag'][2,2], output_array['Y_ang'][0,0], output_array['Y_ang'][0,1], output_array['Y_ang'][0,2], output_array['Y_ang'][1,0], output_array['Y_ang'][1,1], output_array['Y_ang'][1,2], output_array['Y_ang'][2,0], output_array['Y_ang'][2,1], output_array['Y_ang'][2,2], np.log10(output_array['MR1']), np.log10(output_array['MR2']), np.log10(output_array['MR3'])])
data_ouput = np.zeros(data_name.size, dtype=[('var1', 'U10'), ('var2', float)]) #define a two-row array to store data_name and data_value
data_ouput['var1'] = data_name #assign data_name to the first row of data_ouput
data_ouput['var2'] = data_value #assign data_value to the second row of data_ouput
data_ouput = data_ouput.T #here you transpose your data, so to have it in two columns
datafile_path = "/Users/shihyentseng/Documents/mu_tau_project/data_output.dat" #here you select the path where you want to store the ascii file
with open(datafile_path, 'w+') as datafile_id: #here you open the ascii file
    np.savetxt(datafile_id, data_ouput, fmt="%s %.8e") #here the ascii file is written.