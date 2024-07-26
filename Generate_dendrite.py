import numpy as np
from scipy.integrate import solve_ivp
class Dendrite:
    def __init__(self,connection, length_connection, radius_dendrite, params, V_dend_init, r, s):
        self.connection = connection
        self.length = length_connection
        self.radius_dendrite = radius_dendrite
        #params = [param + np.random.normal(0, abs(param) / 2) for param in params]
        self.g_Ca, self.E_Ca, self.g_L, self.E_L, self.g_K, self.E_K, self.g_sd, self.Cm_dend, self.p = params
        self.V_dend_init = V_dend_init 
        self.r_init = r
        self.s_init = s
        
        # Adjust conductances based on length
        self.g_Ca =  radius_dendrite**2 * self.length * np.pi * self.g_Ca
        self.g_L = radius_dendrite**2  * self.length * np.pi * self.g_L
        self.g_K = radius_dendrite**2  * self.length * np.pi * self.g_K
        
        # Adjust capacitance based on length
        self.Cm_dend = radius_dendrite**2 * self.length * np.pi * self.Cm_dend 

    def dend_currents(self, V_soma, V_dend, r, s, num_dendrites):
        I_Ca = self.g_Ca * r**2 * (V_dend - self.E_Ca) 
        I_K = self.g_K * s * (V_dend - self.E_K)
        I_L_dend = self.g_L * (V_dend - self.E_L)
        if num_dendrites == 0:
            p = 1
        else:
            p = num_dendrites
        I_sd = self.g_sd/p * self.radius_dendrite**2 * np.pi * (V_dend - V_soma)  

        return I_L_dend, I_sd, I_Ca, I_K

    def update_gating_variables(self, V_dend, r, s, dt):
        # Rate equations for r (Ca activation)
        alpha_r = 1.6 / (1 + np.exp(-(V_dend - 5) / 14))
        beta_r = 0.02 * (V_dend + 8.5) / (1 - np.exp((V_dend + 8.5) / 5 + 1e-9))  # Avoid division by zero
        tau_r = 1 / (alpha_r + beta_r)
        r_inf = alpha_r / (alpha_r + beta_r)
        dr_dt = (r_inf - r) / tau_r

        # Rate equations for s (K activation)
        alpha_s = 0.016 * (V_dend + 35) / (1 - np.exp(-(V_dend + 35) / 5 + 1e-9))
        beta_s = 0.25 * np.exp(-(V_dend + 45) / 40)
        tau_s = 1 / (alpha_s + beta_s)
        s_inf = alpha_s / (alpha_s + beta_s)
        ds_dt = (s_inf - s) / tau_s
        
        r_next = r + dr_dt * dt
        s_next = s + ds_dt * dt

        # Ensure gating variables remain within [0, 1]
        r_next = np.clip(np.nan_to_num(r_next, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)
        s_next = np.clip(np.nan_to_num(s_next, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)
        return r_next, s_next



    def RK_dend(self, V_soma, V_dend, r,s,  dt, num_dendrites, I_GJ=0):
        def dV_dend_dt(V_dend, r, s, I_GJ):
            I_L_dend, I_sd, I_Ca, I_K = self.dend_currents(V_soma,V_dend, r, s, num_dendrites)
            return - (I_L_dend + I_sd + I_Ca + I_K + I_GJ) / self.Cm_dend

        # k1 values
        k1 = dV_dend_dt(V_dend, r, s, I_GJ)
        r1, s1 = self.update_gating_variables(V_dend, r, s, dt)

        # k2 values
        k2 = dV_dend_dt(V_dend + 0.5 * k1 * dt, r1, s1, I_GJ)
        r2, s2 = self.update_gating_variables(V_dend + 0.5 * k1 * dt, r1, s1, dt)

        # k3 values
        k3 = dV_dend_dt(V_dend + 0.5 * k2 * dt, r2, s2, I_GJ)
        r3, s3 = self.update_gating_variables(V_dend + 0.5 * k2 * dt, r2, s2, dt)

        # k4 values
        k4 = dV_dend_dt(V_dend + k3 * dt, r3, s3, I_GJ)
        r4, s4 = self.update_gating_variables(V_dend + k3 * dt, r3, s3, dt)

        V_dend += (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)
        r = (r1 + 2 * r2 + 2 * r3 + r4) / 6
        s = (s1 + 2 * s2 + 2 * s3 + s4) / 6

        return V_dend, r, s