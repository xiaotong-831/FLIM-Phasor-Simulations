# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 16:23:07 2025

@author: Yuan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad 
from scipy.stats import gaussian_kde

#%% define parameters and define functions/arrays 

# system parameters
gate_width = 1
gate_delay = gate_width
t_max_decay = 25 # ns
t_max = 500 
num_data = 500
t = np.linspace(0, t_max_decay, num_data)
N_photon = 500
rep_rate = 12.5  # ns
w_pulse = 2 * np.pi / rep_rate 

# free & bound & mixed
tau_free = 0.4
tau_bound = 2.5
frac_free = 0.4 # the fraction of the free NADH in mixture 1
frac_free_two = 0.6

tau_mix = frac_free * tau_free + (1 - frac_free) * tau_bound 
tau_mix_two = frac_free_two * tau_free + (1 - frac_free_two) * tau_bound 
# decay_mix = frac_free * np.exp(-t / tau_free) + (1 - frac_free) * np.exp(-t / tau_bound)
# decay_mix_two = frac_free_two * np.exp(-t / tau_free) + (1 - frac_free_two) * np.exp(-t / tau_bound)

# noise parameters 
gaussian_add_noise_std = 2  # std of gaussian assusming a few photons 
gaussian_mul_noise_std = 1/10
num_iterations = 1000

# # only photon noise 
# gaussian_add_noise_std = 0  # std of gaussian assusming a few photons 
# gaussian_mul_noise_std = 0

# Generate the gates
gate_edges = np.arange(0, t_max, gate_delay)
gate_centers = gate_edges + gate_width/2 
num_gates = len(gate_edges)


#%% define functions and calculate clusters 

def phasor_positions(tau):
    g = 1 / (1 + (w_pulse * tau)**2)
    s = w_pulse * tau / (1 + (w_pulse * tau)**2)
    return g, s

# calculate the theoretical positions for free and bound NADH 
g_free_theo, s_free_theo = phasor_positions(tau_free)
g_bound_theo, s_bound_theo = phasor_positions(tau_bound)
g_mix_theo, s_mix_theo = phasor_positions(tau_mix)
g_mix_two_theo, s_mix_two_theo = phasor_positions(tau_mix_two)

# Generate the decays and integrate over each gate window  
# returned array contains elements represent the expected number of photons in each gate 
def calculate_expected_photons(tau, N_total):
    total_integral, _ = quad(lambda t: np.exp(-t/tau), 0, t_max)
    
    # calculate integral for each gate 
    expected = np.zeros(num_gates)
    for i, edge in enumerate(gate_edges):
        gate_integral, _ = quad(lambda t: np.exp(-t/tau), edge, edge + gate_width)
        expected[i] = (gate_integral / total_integral) * N_total 
    return expected 

# generations of free, bound and mixed decays
expected_free = calculate_expected_photons(tau_free, N_photon)
expected_bound = calculate_expected_photons(tau_bound, N_photon)
expected_mix = frac_free * expected_free + (1 - frac_free) * expected_bound  
expected_mix_two = frac_free_two * expected_free + (1 - frac_free_two) * expected_bound

# apply noise    
def noisy_phasor(expected_photons):
    
    poisson_noise = np.random.poisson(expected_photons)
    gaussian_add = np.random.normal(0, gaussian_add_noise_std, size=poisson_noise.shape)
    gaussian_mul = np.random.normal(1, gaussian_mul_noise_std, size=poisson_noise.shape)
    # total_noise = np.clip(gaussian_mul*poisson_noise + gaussian_add, 0, None)
    total_noise = gaussian_mul*poisson_noise + gaussian_add
    
    # phasor calculation
    total = np.sum(total_noise) 
    if total == 0:
        return np.nan, np.nan 
    g = np.sum(total_noise * np.cos(w_pulse*gate_centers)) / total
    s = np.sum(total_noise * np.sin(w_pulse*gate_centers)) / total
    return g, s

# calculate phasor clusters 
phasor_cluster_free = np.array([noisy_phasor(expected_free) for _ in range(num_iterations)])
phasor_cluster_bound = np.array([noisy_phasor(expected_bound) for _ in range(num_iterations)])
phasor_cluster_mix = np.array([noisy_phasor(expected_mix) for _ in range(num_iterations)])
phasor_cluster_mix_two = np.array([noisy_phasor(expected_mix_two) for _ in range(num_iterations)])

# analyze the clouds 
def calculate_cloud_size(phasor_cluster):
    g_std = np.std(phasor_cluster[:, 0]) 
    s_std = np.std(phasor_cluster[:, 1])  
    return np.sqrt(g_std**2 + s_std**2)  
    
def calculate_distance(phasor_cluster, g_theo, s_theo):
    g_mean = np.mean(phasor_cluster[:, 0])
    s_mean = np.mean(phasor_cluster[:, 1])
    return np.sqrt((g_mean - g_theo)**2 + (s_mean - s_theo)**2)    
    
metrics = {
    'Free NADH': {
        'cluster': phasor_cluster_free,
        'theory': (g_free_theo, s_free_theo)
    },
    'Bound NADH': {
        'cluster': phasor_cluster_bound,
        'theory': (g_bound_theo, s_bound_theo)
    },
    'Mixture': {
        'cluster': phasor_cluster_mix,
        'theory': (
            frac_free * g_free_theo + (1 - frac_free) * g_bound_theo,
            frac_free * s_free_theo + (1 - frac_free) * s_bound_theo
        )
    }
} 

# # calculate the size of the cloud and the distance to ground truth 
# for name, data in metrics.items():
#     cluster = data['cluster'][~np.isnan(data['cluster']).any(axis=1)]  # Remove NaNs
#     g_theo, s_theo = data['theory']
    
#     cloud_size = calculate_cloud_size(cluster)
#     distance = calculate_distance(cluster, g_theo, s_theo)
    
#     metrics[name]['cloud_size'] = cloud_size
#     metrics[name]['distance'] = distance
#     print(f"{name}: Cloud size = {cloud_size:.4f}, Distance to theory = {distance:.4f}")


# calculate measurement density using kernel density estimation to colorcode the phasor 
def calculate_density(points): 
    if len(points) == 0:
        return np.array([])
    
    values = np.vstack([points[:, 0], points[:, 1]])
    kernel = gaussian_kde(values)
    density = kernel(values)
    return density 
        
# Calculate densities for each cluster
density_mix = calculate_density(phasor_cluster_mix[~np.isnan(phasor_cluster_mix).any(axis=1)])
density_mix_two = calculate_density(phasor_cluster_mix_two[~np.isnan(phasor_cluster_mix_two).any(axis=1)])    


#%% plot the new phasor trajectory 

# loop over lifetimes to plot new semicircles (NOISE = 0)
lifetimes_array = np.linspace(0.01, 300, 1000)  

g_array = []
s_array = []

def ideal_phasor(expected_photons):
    total = np.sum(expected_photons)
    if total == 0:
        return np.nan, np.nan
    g = np.sum(expected_photons * np.cos(w_pulse * gate_centers)) / total
    s = np.sum(expected_photons * np.sin(w_pulse * gate_centers)) / total
    return g, s

# Loop over lifetimes and simulate expected gated signal
for tau in lifetimes_array:
    expected = calculate_expected_photons(tau, N_photon)
    g, s = ideal_phasor(expected)
    g_array.append(g)
    s_array.append(s)

# Gated (simulated) phasor positions for free and bound NADH
expected_free_clean = calculate_expected_photons(tau_free, N_photon)
expected_bound_clean = calculate_expected_photons(tau_bound, N_photon)
g_free_new, s_free_new = ideal_phasor(expected_free_clean)
g_bound_new, s_bound_new = ideal_phasor(expected_bound_clean)

# _new is the NEW position where the phasor is supposed to be on the distorted trajectory 
g_mix_new = frac_free * g_free_new + (1 - frac_free) * g_bound_new
s_mix_new = frac_free * s_free_new + (1 - frac_free) * s_bound_new

g_mix_two_new = frac_free_two * g_free_new + (1 - frac_free_two) * g_bound_new
s_mix_two_new = frac_free_two * s_free_new + (1 - frac_free_two) * s_bound_new 


#%%

plt.figure(figsize = (10,6))

# draw the universal semicircle
theta = np.linspace(0, np.pi, 1000)  
g_semicircle = 0.5 + 0.5 * np.cos(theta)
s_semicircle = 0.5 * np.sin(theta)
plt.plot(g_semicircle, s_semicircle, color='black')

# plot the theoretical positions of free & bound NADH phasors AND their actual phasor clusters 
plt.scatter(g_free_theo, s_free_theo, color = 'blue', marker='o', s = 120)
# plt.scatter(g_bound_theo, s_bound_theo, color = 'red',marker='o', s = 120)
plt.scatter(phasor_cluster_free[:, 0], phasor_cluster_free[:, 1], color = 'blue')
plt.scatter(phasor_cluster_bound[:, 0], phasor_cluster_bound[:, 1], color = 'red')

# plot the two phasor clusters of the color-coded mixtures 
phasor_cluster_mix_clean = phasor_cluster_mix[~np.isnan(phasor_cluster_mix).any(axis = 1)]
phasor_cluster_mix_two_clean = phasor_cluster_mix_two[~np.isnan(phasor_cluster_mix_two).any(axis = 1)]

# use KDE to calculate density 
values = np.vstack([phasor_cluster_mix_clean[:, 0], phasor_cluster_mix_clean[:, 1]])
values_two = np.vstack([phasor_cluster_mix_two_clean[:, 0], phasor_cluster_mix_two_clean[:, 1]])
kde = gaussian_kde(values)
kde_two = gaussian_kde(values_two)
densities = kde(values)
densities_two = kde_two(values_two)

# to sort the phasor points by density 
# plot low-density points first and high-density points on top 
idx = densities.argsort()  # sort values from lowest to highest 
g_sorted = phasor_cluster_mix_clean[:, 0][idx]
s_sorted = phasor_cluster_mix_clean[:, 1][idx]
dens_sorted = densities[idx]

idx_two = densities_two.argsort()
g_sorted_two = phasor_cluster_mix_two_clean[:, 0][idx_two]
s_sorted_two = phasor_cluster_mix_two_clean[:, 1][idx_two]
dens_sorted_two = densities_two[idx_two]

plt.scatter(g_sorted, s_sorted, c=dens_sorted, cmap='viridis', s=20, edgecolor='none')
plt.colorbar(label='Mixture one')
plt.scatter(g_sorted_two, s_sorted_two, c=dens_sorted_two, cmap='plasma', s=20, edgecolor='none')
plt.colorbar(label='Mixture two')

# draw the line connecting the two theoretical values 
plt.plot([g_free_theo, g_bound_theo], [s_free_theo, s_bound_theo], color = 'orange', alpha=0.8, linewidth=2)
plt.plot([g_free_new, g_bound_new], [s_free_new, s_bound_new], color = 'orange', alpha=0.8, linewidth=2)

plt.scatter(g_mix_new, s_mix_new, color='blue', marker='*', s=120)
# plt.scatter(g_mix_two_new, s_mix_two_new, color='red', marker='*', s=120)


# Calculate and plot the exact theoretical positions of the mixtures
for frac, color, label in zip([frac_free, frac_free_two], 
                              ['blue', 'red'], 
                              [f'Mixture 1 ({frac_free*100:.0f}% free)', 
                              f'Mixture 2 ({frac_free_two*100:.0f}% free)']):
    g_mix = frac * g_free_theo + (1 - frac) * g_bound_theo
    s_mix = frac * s_free_theo + (1 - frac) * s_bound_theo
    plt.scatter(g_mix, s_mix, color=color, marker='*', s=200, label=label)
    
# Mark specific lifetimes on the semicircle
lifetimes_to_mark = [0.5, 1, 2, 3, 4, 6, 10]  # ns

for tau in lifetimes_to_mark:
    g = 1 / (1 + (w_pulse * tau) ** 2)
    s = (w_pulse * tau) / (1 + (w_pulse * tau) ** 2)
    plt.scatter(g, s, color = 'black', s = 20, marker = 'o')
    plt.text(g, s, f"{tau} ns", fontsize = 10, ha='right')
    
# Clean clusters (remove NaNs)
free_clean = phasor_cluster_free[~np.isnan(phasor_cluster_free).any(axis=1)]
bound_clean = phasor_cluster_bound[~np.isnan(phasor_cluster_bound).any(axis=1)]

# Mean gated phasor positions
g_free_gated = np.mean(free_clean[:, 0])
s_free_gated = np.mean(free_clean[:, 1])
g_bound_gated = np.mean(bound_clean[:, 0])
s_bound_gated = np.mean(bound_clean[:, 1])

plt.plot(g_array, s_array, '--', color='green', linewidth=2)

plt.xlabel("G", fontsize=14)
plt.ylabel("S", fontsize=14)
plt.xlim(0,1.2)
plt.ylim(0,0.8)
plt.title(f'Photon Number = {N_photon}, Gate Width = Gate Delay = {gate_width}ns', fontsize=12)
plt.legend(loc="upper right", fontsize=8)
plt.show()


