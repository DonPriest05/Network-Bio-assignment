import numpy as np
from scipy.integrate import solve_ivp
class Soma:

    def __init__(self, params, V_soma_init, size_soma, n_init, h_init, k_init, l_init, q_init):
        self.size_soma = size_soma
        #params = [param + np.random.normal(0, abs(param) / 2) for param in params]
        self.g_Ca, self.g_h, self.E_Ca, self.E_h, self.g_L, self.E_L, self.g_Na, self.E_Na, self.g_K, self.E_K, self.g_sd, self.Cm_soma, self.p = params
        self.V_soma_init = V_soma_init
        self.n_init = n_init
        self.h_init = h_init
        self.k_init = k_init
        self.l_init = l_init
        self.q_init = q_init

        # Adjust conductances based on area
        self.g_Ca = self.size_soma * self.g_Ca
        self.g_L = self.size_soma * self.g_L 
        self.g_K = self.size_soma * self.g_K 
        self.g_h = self.size_soma * self.g_h
        self.g_Na = self.size_soma * self.g_Na
 
        # Adjust capacitance based on area
        self.Cm_soma = self.size_soma * self.Cm_soma




    def m_inf(self, V):
        alpha_m = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10 + 1e-9))  # Avoid division by zero
        
        beta_m = 4 * np.exp(-(V + 65) / 18)
        return alpha_m / (alpha_m + beta_m)
    
    def soma_currents(self, V_soma, V_dend, n, h, k, l, q, radii, num_dendrites):
        I_Ca_soma = self.g_Ca * k**3 * l * (V_soma - self.E_Ca)
        
        I_L_soma = self.g_L * (V_soma - self.E_L)
        I_h = self.g_h * q * (V_soma - self.E_h)
        I_Na_soma = self.g_Na * (self.m_inf(V_soma)**3) * h * (V_soma - self.E_Na)
        I_K_soma = self.g_K * (n**4) * (V_soma - self.E_K)
        if num_dendrites == 0:
            p = 0
        else:
            p = num_dendrites
        I_sd = sum(self.g_sd/p * radii[i]**2*np.pi * (V_soma - Vd) for i,Vd in enumerate(V_dend))
        return I_Ca_soma, I_L_soma, I_h, I_Na_soma, I_K_soma, I_sd
        
    def update_gating_variables(self, V_soma, n, h, k, l, q, dt):
        # Rate equations for n
        alpha_n = 0.01 * (V_soma + 55) / (1 - np.exp(-(V_soma + 55) / 10))
        beta_n = 0.125 * np.exp(-(V_soma + 65) / 80)
        tau_n = 1 / (alpha_n + beta_n)
        n_inf = alpha_n / (alpha_n + beta_n)
        dn_dt = (n_inf - n) / tau_n

        # Rate equations for h
        alpha_h = 0.07 * np.exp(-(V_soma + 65) / 20)
        beta_h = 1 / (1 + np.exp(-(V_soma + 35) / 10))
        h_inf = alpha_h / (alpha_h + beta_h)
        tau_h = 1 / (alpha_h + beta_h)
        dh_dt = (h_inf - h) / tau_h

        # Rate equations for k
        k_inf = 1 / (1 + np.exp(-(V_soma + 61) / 4.2))
        tau_k = 5
        dk_dt = (k_inf - k) / tau_k

        # Rate equations for l
        l_inf = 1 / (1 + np.exp((V_soma + 85.5) / 8.5))
        exp_val1 = np.exp((V_soma + 160) / 30)
        exp_val2 = np.exp((V_soma + 84) / 7.3)
        tau_l = (20 * exp_val1) / (1 + exp_val2) + 35
        dl_dt = (l_inf - l) / tau_l

        # Rate equations for q
        q_inf = 1 / (1 + np.exp((V_soma + 75) / 5.5))
        tau_q = 1 / (np.exp(-0.086 * V_soma - 15.6) + np.exp(0.07 * V_soma - 1.87))
        dq_dt = (q_inf - q) / tau_q
        
        # Update gating variables
        n_next = n + dn_dt * dt
        h_next = h + dh_dt * dt
        k_next = k + dk_dt * dt
        l_next = l + dl_dt * dt
        q_next = q + dq_dt * dt

        # Ensure gating variables remain within [0, 1]
        n_next = np.clip(np.nan_to_num(n_next, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)
        h_next = np.clip(np.nan_to_num(h_next, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)
        k_next = np.clip(np.nan_to_num(k_next, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)
        l_next = np.clip(np.nan_to_num(l_next, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)
        q_next = np.clip(np.nan_to_num(q_next, nan=0.0, posinf=1.0, neginf=0.0), 0, 1)

        return n_next, h_next, k_next, l_next, q_next

    def RK_soma(self, V_soma, V_dend, n, h, k, l, q, I, dt, radii, num_dendrites, I_GJ=0):
        def dV_soma_dt(V_soma, n, h, k, l, q, I_GJ):
            I_Ca_soma, I_L_soma, I_h, I_Na_soma, I_K_soma, I_sd = self.soma_currents(V_soma, V_dend, n, h, k, l, q, radii, num_dendrites)
            return (-(I_Ca_soma + I_L_soma + I_h + I_Na_soma + I_K_soma + I_sd + I_GJ) + I)/self.Cm_soma

        # k1 values
        k1_v = dV_soma_dt(V_soma, n, h, k, l, q, I_GJ)
        n1, h1, k1, l1, q1 = self.update_gating_variables(V_soma, n, h, k, l, q, dt)

        # k2 values
        k2_v = dV_soma_dt(V_soma + 0.5 * k1_v * dt, n1, h1, k1, l1, q1, I_GJ)
        n2, h2, k2, l2, q2 = self.update_gating_variables(V_soma + 0.5 * k1_v * dt, n1, h1, k1, l1, q1, dt)

        # k3 values
        k3_v = dV_soma_dt(V_soma + 0.5 * k2_v * dt, n2, h2, k2, l2, q2, I_GJ)
        n3, h3, k3, l3, q3 = self.update_gating_variables(V_soma + 0.5 * k2_v * dt, n2, h2, k2, l2, q2, dt)

        # k4 values
        k4_v = dV_soma_dt(V_soma + k3_v * dt, n3, h3, k3, l3, q3, I_GJ)
        n4, h4, k4, l4, q4 = self.update_gating_variables(V_soma + k3_v * dt, n3, h3, k3, l3, q3, dt)

        V_soma_next = V_soma + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) * (dt / 6)

        n = (n1 + 2 * n2 + 2 * n3 + n4) / 6
        h = (h1 + 2 * h2 + 2 * h3 + h4) / 6
        k = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        l = (l1 + 2 * l2 + 2 * l3 + l4) / 6
        q = (q1 + 2 * q2 + 2 * q3 + q4) / 6

        return V_soma_next, n, h, k, l, q