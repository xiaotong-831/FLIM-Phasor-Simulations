# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:17:55 2025
@author: Yuan
"""

# Step 0: plot the decay & gates for visualization purposes (GW = 500ps, GD = 500ps; gate starts at 0) 
# Step 1: calculate detected photons in each gate -> display the curve
# Step 2: add noise (Poisson + Gaussian) -> display the results 
# Step 3: calculate errors from fitting 
# Step 4: repeat step 2-3
# Step 5: add weighted lifetime method 
# Step 6: phasor analysis 

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy import signal

#%% define parameters and initialize functions/arrays 

# Parameters
gate_width = 1  # ns
gate_delay = gate_width
tau_true = 3.0
# photon_counts_array = np.array([100, 250, 500, 750, 1000, 2000, 3000, 4000, 5000])*np.sqrt(2)
photon_counts_array = np.array([100, 250, 500, 750, 1000, 2000, 3000, 4000, 5000])
t_max = 25 # ns
num_data = 500
t = np.linspace(0, t_max, num_data)
num_iterations = 100
gaussian_add_noise_std = 2  # std of gaussian assusming a few photons 
gaussian_mul_noise_std = 1/100

gw_ratio = gate_width /tau_true

f_pulse = 80e6 
# w_pulse = 2 * np.pi * f_pulse   
rep_rate = 12.5  # ns
w_pulse = 2 * np.pi / rep_rate 

# Simulate the fluorescence decay
decay_curve = np.exp(-t / tau_true)
decay_curve_norm = decay_curve / np.sum(decay_curve)
decay_curve = decay_curve_norm  # Normalized decay curve

# Generate the gates
gate_edges = np.arange(0, t_max, gate_delay)
gate_centers = gate_edges + gate_width/2 

# Initialize arrays
mean_relative_errors_fit = []  # Relative errors for curve fitting
std_relative_errors_fit = []
mean_relative_errors_weighted = []  # Relative errors for weighted lifetime
mean_relative_errors_phasor = []
std_relative_errors_weighted = []
mean_fitted_taus = []
mean_weighted_taus = [] 
mean_phasor_taus = [] 
# calculate lifetimes from the phasors
phasor_lifetimes = {N: [] for N in photon_counts_array}
mean_phasor_lifetimes = {N: np.mean(phasor_lifetimes[N]) for N in photon_counts_array}

# Define functions 
def decay_func(t, tau):
    return np.exp(-t / tau)

# Define decay curve for fitting
def exp_decay(t, A, tau):
    return A * np.exp(-t / tau)

def calculate_phasor(decay, omega, t):
    integral_total = np.sum(decay)
    if integral_total == 0:  
        return np.nan, np.nan    
    g = np.sum(decay * np.cos(omega * t)) / integral_total
    s = np.sum(decay * np.sin(omega * t)) / integral_total
    # print(f"cos(omega*t): {np.cos(omega * t)}")  # Debugging
    # print(f"sin(omega*t): {np.sin(omega * t)}")  # Debugging
    # print(f"g: {g}, s: {s}")  # Debugging
    return g, s

# Store phasor clusters
phasor_g_all = {N: [] for N in photon_counts_array}
phasor_s_all = {N: [] for N in photon_counts_array}


#%% Loop over different photon numbers
for N_photon in photon_counts_array:
    N_detected = []
    fitted_taus = []
    weighted_taus = []
    phasor_taus = []

    # Calculate expected photon count using integration within each gate
    for edge in gate_edges:
        integral, _ = quad(decay_func, edge, edge + gate_width, args=(tau_true,))
        photon_count = (integral / np.sum(decay_curve)) * N_photon
        N_detected.append(photon_count)

    # Repeat for noise analysis
    for _ in range(num_iterations):
        noise_N_detected = np.random.poisson(N_detected)  # poisson noise 
        gaussian_add_noise = np.random.normal(0, gaussian_add_noise_std, size=noise_N_detected.shape)   # gaussian noise 
        gaussian_mul_noise = np.random.normal(1, gaussian_mul_noise_std, size=noise_N_detected.shape)   # amplification niose 
        noise_N_detected_total = gaussian_mul_noise * noise_N_detected + gaussian_add_noise
        # noise_N_detected_total = np.clip(noise_N_detected_total, 0, None) # should not clip it because we want to maintain the negative noise 
        
        try:
            # Curve fitting and weighted lifetime calculations
            params, covariance = curve_fit(exp_decay, gate_centers, noise_N_detected_total, p0=[max(noise_N_detected_total), tau_true])
            fitted_tau = params[1]
            fitted_taus.append(fitted_tau)

            weighted_tau = np.sum(gate_centers * noise_N_detected_total) / np.sum(noise_N_detected_total)
            weighted_taus.append(weighted_tau)

            # Phasor analysis
            g, s = calculate_phasor(noise_N_detected_total, w_pulse, gate_centers)
            phasor_tau = (1 / w_pulse) * (s / g)
            phasor_taus.append(phasor_tau)
            
            filtered_g = signal.medfilt2d(g, kernel_size = 3)
            filtered_s = signal.medfilt2d(s, kernel_size = 3)
            
            
            if not np.isnan(g) and not np.isnan(s):  # Only append valid phasor values
                phasor_g_all[N_photon].append(filtered_g)
                phasor_s_all[N_photon].append(filtered_s)
        

            # if not np.isnan(g) and not np.isnan(s):  # Only append valid phasor values
            #     phasor_g_all[N_photon].append(g)
            #     phasor_s_all[N_photon].append(s)

        except RuntimeError:
            continue

    # Analyze the results (curve fitting, weighted lifetime, phasor)
    mean_fitted_tau = np.mean(fitted_taus)
    std_fitted_tau = np.std(fitted_taus)
    relative_error_fit = (std_fitted_tau / mean_fitted_tau) * 100
    mean_relative_errors_fit.append(relative_error_fit)
    mean_fitted_taus.append(mean_fitted_tau)

    mean_weighted_tau = np.mean(weighted_taus)
    std_weighted_tau = np.std(weighted_taus)
    relative_error_weighted = (std_weighted_tau / mean_weighted_tau) * 100
    mean_relative_errors_weighted.append(relative_error_weighted)
    mean_weighted_taus.append(mean_weighted_tau)

    mean_phasor_tau = np.mean(phasor_taus)
    std_phasor_tau = np.std(phasor_taus)
    relative_error_phasor = (std_phasor_tau / mean_phasor_tau) * 100
    mean_relative_errors_phasor.append(relative_error_phasor)
    mean_phasor_taus.append(mean_phasor_tau)


#%% PLOT

# DEBUGGING
plt.figure(figsize=(8, 6))
# plt.plot(gate_centers, noise_N_detected, label="Poisson Noise")
# plt.plot(gate_centers, gaussian_add_noise, label="Gaussian Noise")
# plt.plot(gate_centers, gaussian_mul_noise, label="Amplification Noise")
# # plt.xlim(0, 6)
# plt.legend()

# Plot 6: Relative errors from curve fitting and weighted lifetime
plt.figure(figsize=(8, 6))
plt.errorbar(photon_counts_array, mean_relative_errors_fit, fmt='o-', label="Curve Fitting")
plt.errorbar(photon_counts_array, mean_relative_errors_weighted, fmt='o-', label="Weighted Lifetime")
plt.errorbar(photon_counts_array, mean_relative_errors_phasor, fmt='o-', label="Phasor Analysis")
plt.xlabel("Photon Count", fontsize=14)
plt.ylabel("Relative Error (%)", fontsize=14)
plt.title("Relative Error vs. Photon Count", fontsize=14)
plt.legend()
plt.show()

# Plot 7: Mean Fitted Lifetime Curve Fitting vs Weighted Lifetime
plt.figure(figsize=(8, 6))
# plt.errorbar(photon_counts_array, mean_fitted_taus, yerr = std_fitted_taus, fmt='o-', label="Curve Fitting")
# plt.errorbar(photon_counts_array, mean_weighted_taus,yerr = std_weighted_taus, fmt='s-', label="Weighted Lifetime")
plt.errorbar(photon_counts_array, mean_fitted_taus, fmt='o-', label="Curve Fitting")
plt.errorbar(photon_counts_array, mean_weighted_taus, fmt='s-', label="Weighted Lifetime")
# mean_phasor_lifetimes_list = np.array([mean_phasor_lifetimes[N] for N in photon_counts_array])
plt.errorbar(photon_counts_array, mean_phasor_taus, fmt='o-', label="Phasor Analysis")
plt.axhline(y=tau_true, color='r', linestyle='--', label="True Lifetime")  # Reference line for true lifetime
plt.xlabel("Photon Count", fontsize=14)
plt.ylabel("Mean Fitted Lifetime (ns)", fontsize=14)
plt.title("Mean Fitted Lifetime vs. Photon Count", fontsize=14)
plt.legend()
plt.show()

# Plot the phasor
plt.figure(figsize=(8, 4))
# Draw the universal semicircle
theta = np.linspace(0, np.pi, 1000)  
g_semicircle = 0.5 + 0.5 * np.cos(theta)
s_semicircle = 0.5 * np.sin(theta)
plt.plot(g_semicircle, s_semicircle, color='black')

# Color map for photon count groups
colors = plt.cm.viridis(np.linspace(0, 1, len(photon_counts_array)))

# Scatter plot with color-coded clusters
for i, N_photon in enumerate(photon_counts_array):
    if len(phasor_g_all[N_photon]) > 0:  # Only plot if data exists
        plt.scatter(phasor_g_all[N_photon], phasor_s_all[N_photon], 
                    color=colors[i], s = 12, label=f"{N_photon} photons")

# Mark specific lifetimes on the semicircle
lifetimes_to_mark = [0.5, 1, 2, 3, 4, 6, 10]  # ns

for tau in lifetimes_to_mark:
    g = 1 / (1 + (w_pulse * tau) ** 2)
    s = (w_pulse * tau) / (1 + (w_pulse * tau) ** 2)
    plt.scatter(g, s, color = 'black', s = 20, marker = 'o')
    plt.text(g, s, f"{tau} ns", fontsize = 10, ha='right')
    
# plt.colorbar(label="Lifetime (ns)")
plt.xlabel("G", fontsize=14)
plt.ylabel("S", fontsize=14)
# plt.xlim[0, 1.2]
plt.title(f'Gate Width = {gate_width}ns, G tau = {tau_true}ns', fontsize=12)

# plt.legend(loc="upper right", fontsize=8, frameon=False)
# plt.legend(loc="upper right", bbox_to_anchor=(1, 1.3), fontsize=8, frameon=False)
plt.show()


#%%

# # Plot 1: Fluorescence decay and gates
# plt.figure(figsize=(8, 6))
# plt.plot(t, decay_curve, label="Fluorescence decay", linewidth=2, color="black")
# for edge in gate_edges:
#     mask = (t >= edge) & (t <= edge + gate_width)
#     plt.fill_between(t[mask], 0, decay_curve[mask], color='red', alpha=0.3)
# plt.xlabel("Time (ns)", fontsize=14)
# plt.ylabel("Intensity", fontsize=14)
# plt.title("Decay and gating visualization", fontsize=14)
# # plt.legend()
# plt.show()

# # Plot 2: Expected photon counts
# plt.figure(figsize=(8, 6))
# plt.bar(gate_centers, N_detected, width=gate_width*0.8, color="blue", alpha=0.7, label="Detected photons")
# plt.xlabel("Time (ns)", fontsize=14)
# plt.ylabel("Photon counts", fontsize=14)
# plt.title("Expected photon counts per gate", fontsize=14)
# plt.legend()
# plt.show()

# # Plot 3: Noisy photon count with the fitted exp curve
# plt.figure(figsize=(8, 6))
# plt.bar(gate_centers, noise_N_detected, width=gate_width*0.8, color="green", alpha=0.7, label="Noisy photons")
# plt.plot(gate_centers, exp_decay(np.array(gate_centers), *params), color="red", label="Fitted curve")
# plt.xlabel("Time (ns)", fontsize=14)
# plt.ylabel("Photon counts", fontsize=14)
# plt.title("Noisy Photon Counts and Fitted Curve")
# plt.legend()
# plt.show()

# # Plot 4&5: Distribution of the fitted lifetime (from curve fitting)
# plt.figure(figsize=(8, 6))
# plt.hist(fitted_taus, bins=20, color="blue", alpha=0.7)
# plt.axvline(tau_true, color="red", linestyle="--", label="True lifetime")
# plt.xlabel("Fitted Lifetime (ns)", fontsize=14)
# plt.ylabel("Frequency", fontsize=14)
# plt.title("Distribution of Lifetimes from Curve Fitting")
# plt.legend()
# plt.show()

# # distribution of lifetime from weighted lifetime 
# plt.figure(figsize=(8, 6))
# plt.hist(weighted_taus, bins=20, color="blue", alpha=0.7)
# plt.axvline(tau_true, color="red", linestyle="--", label="True lifetime")
# plt.xlabel("Fitted Lifetime (ns)", fontsize=14)
# plt.ylabel("Frequency", fontsize=14)
# plt.title("Distribution of Lifetimes from Weighted Lifetime")
# plt.legend()
# plt.show()



