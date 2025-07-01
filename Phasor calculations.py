# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 11:31:00 2025

@author: Yuan
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff 
from scipy.stats import gaussian_kde

#%% experimental data 

# stack = tiff.imread(r"C:\Users\yuan\Desktop\CZI project\FLUTE\Dataset\Fluorescein_Embryo.tif")
# stack = tiff.imread(r"C:\Users\yuan\Desktop\Data\Experiment 2025\5-6-2025_images\Subtracted_Fluorescence_GW1GD1.tif")  
# stack = tiff.imread(r"C:\Users\yuan\Desktop\Data\Trimscope result\2025-1-8 trimscope\Beads 100diluted\FOV1.tif")
# stack = tiff.imread(r"C:\Users\yuan\Desktop\Data\Experiment 2025\5-23-2025 images\ExperimentA_photon_effect_HRI850_GW1ns\subtracted_FL_EOM800.tif")
stack = tiff.imread(r"C:\Users\yuan\Desktop\Data\Experiment 2025\5-23-2025 images\ExperimentB_photon_amp_noise\subtracted_FL_HRI500_EOM500.tif")

# # loop over a stack 
# for i, frame in enumerate(stack):
#     plt.imshow(frame, cmap='gray')
#     plt.title(f"Frame {i}")
#     plt.axis('off')
#     plt.show() 


#%% 

# parameters 
n_gates, H, W = stack.shape
# gate_width = 0.221 # bin width of FLUTE dataset 
# gate_width = 0.5 # trimscope 
gate_width = 1.0  # light sheet setup 
t_start = 0.0
t_k = np.array([t_start + gate_width * (k + 0.5) for k in range(n_gates)])
f = 80e6
omega = 2*np.pi*f*1e-9
I = stack.reshape(n_gates, -1)  # shape: (n_gates, N_pixels)

cos_t = np.cos(omega * t_k)[:, np.newaxis]  # shape: (n_gates, 1)
sin_t = np.sin(omega * t_k)[:, np.newaxis]

# calculate phasor components 
numerator_g = np.sum(I * cos_t, axis=0)
numerator_s = np.sum(I * sin_t, axis=0)
denominator = np.sum(I, axis=0) + 1e-10  

# apply an intensity threshold 
I_sum = np.sum(I, axis=0)
intensity_threshold = np.percentile(I_sum, 80) * 1  # 5% of max intensity, adjustable
valid_mask = I_sum > intensity_threshold

# Calculate g and s for phasor 
g = numerator_g[valid_mask] / denominator[valid_mask]
s = numerator_s[valid_mask] / denominator[valid_mask]

# g_valid = g[valid_mask]
# s_valid = s[valid_mask]

# # reshape back to image 
# g_img = g.reshape(H, W)
# s_img = s.reshape(H, W)


#%% Calibration 

tau_ref = 0.0
# calibration_data = tiff.imread(r"C:\Users\yuan\Desktop\CZI project\FLUTE\Dataset\Fluorescein_Embryo.tif")
calibration_data = tiff.imread(r"C:\Users\yuan\Desktop\Data\Experiment 2025\5-6-2025_images\subtracted_scattered.tif")  
# calibration_data = tiff.imread(r"C:\Users\yuan\Desktop\Data\Trimscope result\2025-1-8 trimscope\Fluorescein\Document1_13-45-40_DC-TCSPC_C0.ome.tif")

# reshape 
# n_gates, Hc, Wc = calibration_data.shape
Ic = calibration_data.reshape(n_gates, -1)  # (n_gates, N_pixels)

# calculate measured phasor BEFORE calibration from the imported calibration data 
numerator_g_c = np.sum(Ic * cos_t, axis=0)
numerator_s_c = np.sum(Ic * sin_t, axis=0)
denominator_c = np.sum(Ic, axis=0) + 1e-10  # avoid divide by zero

# apply intensity threshold to calibration data too
I_sum_c = np.sum(Ic, axis=0)
intensity_threshold_c = np.percentile(I_sum_c, 80) * 1
valid_mask_c = I_sum_c > intensity_threshold_c

# calculate g and s for the calibration data (before calibration)
g_calib = numerator_g_c[valid_mask_c] / denominator_c[valid_mask_c]
s_calib = numerator_s_c[valid_mask_c] / denominator_c[valid_mask_c]
g_meas = np.mean(g_calib)
s_meas = np.mean(s_calib)

# calculate actual phasor position of calibration dataset (before being calibrated)
phi_meas = np.arctan2(s_meas, g_meas)
mod_meas = np.sqrt(g_meas**2 + s_meas**2)

# calculate theoretical phasor position 
g_theo = 1 / (1 + (omega*tau_ref)**2)
s_theo = (omega*tau_ref) / (1 + (omega*tau_ref)**2)
phi_theo = np.arctan2(s_theo, g_theo)
mod_theo = np.sqrt(g_theo**2 + s_theo**2)

# calculate correction factors
delta_phi = phi_theo - phi_meas
delta_mod = mod_theo / mod_meas

# apply corrections to calibration data to check if calibration works
phi_corrected_cali = phi_meas + delta_phi
mod_corrected_cali = mod_meas * delta_mod  
g_corrected_cali = mod_corrected_cali * np.cos(phi_corrected_cali)
s_corrected_cali = mod_corrected_cali * np.sin(phi_corrected_cali)

# apply corrections to experimental data 
phi_data = np.arctan2(s, g)
mod_data = np.sqrt(g**2 + s**2)
phi_corrected = phi_data + delta_phi
mod_corrected = mod_data*delta_mod  # corrected experimental phase and modulation 
g_corrected = mod_corrected * np.cos(phi_corrected)
s_corrected = mod_corrected * np.sin(phi_corrected)


#%% add median filters 

# 3x3 convolutional median filter on g and s







#%% plot phasors
 
plt.figure(figsize=(6, 6))
theta = np.linspace(0, np.pi, 1000)  
g_semicircle = 0.5 + 0.5 * np.cos(theta)
s_semicircle = 0.5 * np.sin(theta)
plt.plot(g_semicircle, s_semicircle, color='black')

# add lifetime marks on the semicircle 
lifetimes_to_mark = [0.5, 1, 2, 3, 4, 6, 10]  # ns

for tau in lifetimes_to_mark:
    g = 1 / (1 + (omega * tau) ** 2)
    s = (omega * tau) / (1 + (omega * tau) ** 2)
    plt.scatter(g, s, color = 'black', s = 12, marker = 'o')
    plt.text(g, s, f"{tau} ns", fontsize = 8, ha='right')
    
# color code phasor density 
phasors = np.vstack([g_corrected, s_corrected]) # turn to 2D array 
# perform KDE 
kde = gaussian_kde(phasors)(phasors)
# sort phasors by density for visualization purpose 
idx = kde.argsort()
g_sorted = g_corrected[idx]
s_sorted = s_corrected[idx]
kde_sorted = kde[idx]    
# KDE color-coded scatter plot
plt.scatter(g_sorted, s_sorted, c=kde_sorted, s=4, cmap='viridis', alpha=0.8, edgecolors='none')
# plt.scatter(g_corrected, s_corrected, s=1, alpha=0.3, c='blue')
# plt.colorbar()

plt.xlabel("g")
plt.ylabel("s")
# plt.title("Phasor Plot ")
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 0.65)
plt.gca().set_aspect('equal')
# plt.legend()
plt.show()

# check calibration phasors 
plt.figure(figsize=(6, 6))
theta = np.linspace(0, np.pi, 1000)  
g_semicircle = 0.5 + 0.5 * np.cos(theta)
s_semicircle = 0.5 * np.sin(theta)
plt.plot(g_semicircle, s_semicircle, color='black')

lifetimes_to_mark = [0.5, 1, 2, 3, 4, 6, 10]  # ns

for tau in lifetimes_to_mark:
    g = 1 / (1 + (omega * tau) ** 2)
    s = (omega * tau) / (1 + (omega * tau) ** 2)
    plt.scatter(g, s, color = 'black', s = 12, marker = 'o')
    plt.text(g, s, f"{tau} ns", fontsize = 8, ha='right')
    
plt.scatter(g_theo, s_theo, color='red', s=60, marker='o', label='Theoretical position of calibration sample')
plt.scatter(g_meas, s_meas, color='red', s=80, marker='*', label='Calibration sample phasor BEFORE calibration')  # these are phasors of calibration dataset before the calibration 
# plt.scatter(g_calib, s_calib, s=1, alpha=0.2, label='Phasor of calibration sample BEFORE correction')
plt.scatter(g_corrected_cali, s_corrected_cali, s=80, color = 'blue', marker = 'x', label='Calibration sample phasor AFTER correction')

plt.xlabel('g')
plt.ylabel('s')
plt.legend(loc="upper right", fontsize=6)
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 0.65)
plt.gca().set_aspect('equal')
plt.show()


# # check intensity histogram 
# plt.hist(I_sum, bins=100, log=True)
# plt.axvline(intensity_threshold, color='red', label=f'Threshold = {np.round(intensity_threshold, 2)}%')
# plt.title("Intensity Histogram")
# plt.xlabel("Total intensity across gates")
# plt.ylabel("Pixel count (log)")
# plt.legend()
# plt.show()

# # plt.figure(figsize=(5, 5))
# plt.scatter(g, s, s=1, alpha=0.2)
# plt.xlabel("g")
# plt.ylabel("s")
# plt.title("Phasor Plot")
# plt.xlim(0, 1)
# plt.ylim(0, 0.5)
# plt.gca().set_aspect('equal')
# plt.show()

# # #intensity threshold masking 
# valid_mask_img = valid_mask.reshape(H, W)
# plt.imshow(valid_mask_img, cmap='gray')
# plt.title("Pixels used for phasor plot")
# plt.axis('off')
# plt.show()

# # Only use valid pixels
# tau_phi = (1 / (2 * np.pi * f)) * np.arctan2(s_valid, g_valid)

# # Fill an empty map with NaN, then add valid values
# tau_map = np.full(H * W, np.nan)
# tau_map[valid_mask] = tau_phi
# tau_map = tau_map.reshape(H, W)

# plt.imshow(tau_map, cmap='inferno')
# plt.colorbar(label='Phase Lifetime (ns)')
# plt.title("Lifetime Map")
# plt.axis('off')
# plt.show()

