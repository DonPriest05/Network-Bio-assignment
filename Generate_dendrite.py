import numpy as np

class Dendrite:
    def __init__(self, params, V_dend_init, r, s):
        self.g_Ca, self.E_Ca, self.g_L, self.E_L, self.g_K, self.E_K, self.g_sd, self.Cm_dend, self.p = params
        self.V_dend_init = V_dend_init
        self.r_init = r
        self.s_init = s

    def dend_currents(self, V_soma, V_dend, r, s):
        I_Ca = self.g_Ca * r**2 * (V_dend - self.E_Ca) 
        I_K = self.g_K * s * (V_dend - self.E_K)
        I_L_dend = self.g_L * (V_dend - self.E_L)
        I_sd = self.g_sd / (1 - self.p) * (V_soma - V_dend)  
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
        r_next = np.clip(r_next, 0, 1)
        s_next = np.clip(s_next, 0, 1)

        return r_next, s_next

    def RK_dend(self, V_soma, V_dend, r, s, dt):
        def dV_dend_dt(V_dend, r, s):
            I_L_dend, I_sd, I_Ca, I_K = self.dend_currents(V_soma, V_dend, r, s)
            return (- (I_L_dend + I_sd + I_Ca + I_K)) / self.Cm_dend

        # k1 values
        k1 = dV_dend_dt(V_dend, r, s)
        r1, s1 = self.update_gating_variables(V_dend, r, s, dt)

        # k2 values
        k2 = dV_dend_dt(V_dend + 0.5 * k1 * dt, r1, s1)
        r2, s2 = self.update_gating_variables(V_dend + 0.5 * k1 * dt, r1, s1, dt)

        # k3 values
        k3 = dV_dend_dt(V_dend + 0.5 * k2 * dt, r2, s2)
        r3, s3 = self.update_gating_variables(V_dend + 0.5 * k2 * dt, r2, s2, dt)

        # k4 values
        k4 = dV_dend_dt(V_dend + k3 * dt, r3, s3)
        r4, s4 = self.update_gating_variables(V_dend + k3 * dt, r3, s3, dt)

        V_dend_next = V_dend + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)
        r = (r1 + 2 * r2 + 2 * r3 + r4) / 6
        s = (s1 + 2 * s2 + 2 * s3 + s4) / 6

        return V_dend_next, r, s
