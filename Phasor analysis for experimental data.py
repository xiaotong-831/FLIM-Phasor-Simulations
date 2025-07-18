# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 11:31:00 2025
@author: Yuan
"""


# This code is to plot phasors and analyze them with filters of 
# experimental data we acquire from the microscopy systems 
# add functions to the phasor analysis (filters, cursors, .etc)

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff 
from scipy.stats import gaussian_kde
from scipy.signal import medfilt
from matplotlib.widgets import EllipseSelector
from matplotlib.patches import Ellipse 


#%% import FLIM data

# FLUTE dataset 
# tau_ref = 4.0
# calibration_data = tiff.imread(r"C:\Users\yuan\Desktop\FLUTE\Dataset\Fluorescein_Embryo.tif")
# stack = tiff.imread(r"C:\Users\yuan\Desktop\FLUTE\Dataset\Fluorescein_Embryo.tif")

# light sheet data on fluorescent beads 
# tau_ref = 0.0
# calibration_data = tiff.imread(r"C:\Users\yuan\Desktop\Data\Experiment 2025\5-6-2025_images\subtracted_scattered.tif")  
# stack = tiff.imread(r"C:\Users\yuan\Desktop\Data\Experiment 2025\5-6-2025_images\Subtracted_Fluorescence_GW1GD1.tif")  

tau_ref = 4.0
calibration_data = tiff.imread(r"C:\Users\yuan\Desktop\Data\Trimscope data\2025-07-15-FLIM-SNR-Convallaria\01-Fluoresceine\01-FluoresceineDC-TCSPC_C0.ome.tif")
stack = tiff.imread(r"C:\Users\yuan\Desktop\Data\Trimscope data\2025-07-15-FLIM-SNR-Convallaria\02-Convallaria\Add-16DC-TCSPC_C0.ome.tif")

# # loop over a stack to check 
# for i, frame in enumerate(stack):
#     plt.imshow(frame, cmap='gray')
#     plt.title(f"Frame {i}")
#     plt.axis('off')
#     plt.show() 


#%% parameters initialization and calculation
n_gates, H, W = stack.shape
N_T = stack.shape[0]
gate_width = 0.221   # bin width of trimscopr 
# gate_width = 0.5   # upright 
# gate_width = 1.0  # light sheet setup 
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
intensity_threshold = np.percentile(I_sum, 90) * 1  
valid_mask = (I_sum > intensity_threshold)
valid_intensity = I_sum[valid_mask]

# g and s without intensity threshold
g_no_thre = numerator_g / denominator
s_no_thre = numerator_s / denominator

# Calculate g and s for phasor    
g = numerator_g[valid_mask] / denominator[valid_mask]
s = numerator_s[valid_mask] / denominator[valid_mask]

# g_valid = g[valid_mask]
# s_valid = s[valid_mask]

# # reshape back to image 
# g_img = g.reshape(H, W)
# s_img = s.reshape(H, W)


#%% Calibration 

n_gates, Hc, Wc = calibration_data.shape
Ic = calibration_data.reshape(n_gates, -1)  # (n_gates, N_pixels)

# calculate theoretical phasor position 
g_theo = 1 / (1 + (omega*tau_ref)**2)
s_theo = (omega*tau_ref) / (1 + (omega*tau_ref)**2)
phi_theo = np.arctan2(s_theo, g_theo)
mod_theo = np.sqrt(g_theo**2 + s_theo**2)

# calculate phasor BEFORE calibration 
# from the imported calibration data (wrong phasor)
numerator_g_c = np.sum(Ic * cos_t, axis=0)
numerator_s_c = np.sum(Ic * sin_t, axis=0)
denominator_c = np.sum(Ic, axis=0) + 1e-10  # avoid divide by zero
# apply intensity threshold to calibration data
I_sum_c = np.sum(Ic, axis=0)
intensity_threshold_c = np.percentile(I_sum_c, 80) * 1
valid_mask_c = I_sum_c > intensity_threshold_c
# calculate g and s for the calibration data (before calibration)
g_calib = numerator_g_c[valid_mask_c] / denominator_c[valid_mask_c]
s_calib = numerator_s_c[valid_mask_c] / denominator_c[valid_mask_c]
g_meas = np.mean(g_calib)
s_meas = np.mean(s_calib)
# calculate actual phasor position of calibration dataset (before calibration) 
phi_meas = np.arctan2(s_meas, g_meas)
mod_meas = np.sqrt(g_meas**2 + s_meas**2)

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

# tau_phi =  (1/omega) * (s_corrected/g_corrected)
# tau_mod =  (1/omega) * np.sqrt(((1/(g_corrected**2 + s_corrected**2))-1))


#%% Filters 

######################## MEDIAN FILTERS #######################

# Add median filter from scipy library (med_filt) 
# I_array = I.flatten()
# g_corrected_array = np.array(g_corrected.flatten() , dtype=np.float64)
# s_corrected_array = np.array(s_corrected.flatten(), dtype=np.float64)
# medfilt_g = g_corrected_array.copy()
# medfilt_s = s_corrected_array.copy()

# # I don't think it's looping here 
# num_filters = 3
# for i in range(num_filters):
#     med_filt_g = medfilt(medfilt_g, kernel_size=3)
#     med_filt_s = medfilt(medfilt_s, kernel_size=3)
    
######################## PHASORPY #######################
# Median filter from PhasorPy 
from phasorpy.datasets import fetch
from phasorpy.io import signal_from_imspector_tiff
from phasorpy.phasor import (
    phasor_calibrate,
    phasor_filter_median,
    phasor_filter_pawflim,
    phasor_from_signal,
    phasor_threshold,
)
from phasorpy.plot import plot_image, plot_phasor

###################### Median filter from PhasorPy #####################################
# Extracting the FFT values of the signal and the reference signal
# signal is the raw data imported (stack here) so not thresholded 
# mean, real, imag = phasor_from_signal(I, axis=0)

# mean_filtered, real_filtered, imag_filtered = phasor_filter_median(
#     mean, real, imag, repeat=3, size=3)

# # apply threshold after filtering 
# mean_filtered, real_filtered, imag_filtered = phasor_threshold(
#     mean_filtered, real_filtered, imag_filtered, mean_min=1)

# # plot unfiltered and filtered phasors 
# plot_phasor(
#     *phasor_threshold(mean, real, imag, mean_min=1)[1:],
#     # frequency=frequency,
#     title='Unfiltered phasor coordinates (median filter)',
# )

# plot_phasor(
#     real_filtered,
#     imag_filtered,
#     # frequency=frequency,
#     title='Median-filtered phasor coordinates (3x3 kernel, 3 reptitions)',
# )

########################## Wavelet filter (pawFLIM) #########################
harmonic = [1, 2]
mean_wl, real_wl, imag_wl = phasor_from_signal(I, axis=0, harmonic=harmonic)

mean_filtered_wl, real_filtered_wl, imag_filtered_wl = phasor_threshold(
    *phasor_filter_pawflim(mean_wl, real_wl, imag_wl, harmonic=harmonic), mean_min=1)

mean_filtered_wl, real_filtered_wl, imag_filtered_wl = phasor_filter_pawflim(
    mean_wl, real_wl, imag_wl, harmonic=harmonic, sigma=5, levels=5)


# plot unfiltered and filtered phasors 
plot_phasor(
    *phasor_threshold(mean_wl, real_wl, imag_wl, mean_min=1)[1:],
    # frequency=frequency,  
    title='Unfiltered phasor coordinates (wavelet filter)',
)

plot_phasor(
    real_filtered_wl[0],
    imag_filtered_wl[0],
    # frequency=frequency,
    title='pawFLIM-filtered phasor coordinates (sigma=5, levels=5)',
)


#%% phasor image mapping / cursors 

# Here we want to be able to select regions on the phasor plots and trace back 
# to the corresponding pixels in the image 

from phasorpy.color import CATEGORICAL
from phasorpy.cursors import (
    mask_from_circular_cursor,
    # mask_from_elliptic_cursor,
    # mask_from_polar_cursor,
    pseudo_color,
)
from phasorpy.datasets import fetch
from phasorpy.io import signal_from_lsm
from phasorpy.phasor import phasor_from_signal, phasor_threshold
from phasorpy.plot import PhasorPlot, plot_image

# Use circular cursors to mask regions of interest in the phasor space:
cursors_real = [0.4]
cursors_imag = [0.4]
radius = [0.2]

circular_mask = mask_from_circular_cursor(
    # g_corrected, s_corrected, 
    g_no_thre, g_no_thre, 
    cursors_real, cursors_imag, radius=radius,
)

# the phasors here in the circular mask are not threshold
# circular mask is a 1D array 
circular_mask = circular_mask.flatten()
circular_mask_valid = circular_mask[valid_mask]

# plot 
plot = PhasorPlot(allquadrants=True, title='Circular cursors')
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


# masked image is intensity thresholded image multiplies the circular mask 

# masked_image = np.reshape(masked_intensity*I_sum,(H,W))
masked_image = valid_intensity * circular_mask_valid
masked_image_full = np.zeros_like(valid_mask, dtype = float)
masked_image_full[valid_mask] = masked_image 
masked_image_2D = masked_image_full.reshape(H, W)


# Draw the circle 
circle = plt.Circle(
    (cursors_real[0], cursors_imag[0]),
    radius[0],
    edgecolor='blue',
    facecolor='none',
    linewidth=1,
    linestyle='-'
)
plt.gca().add_patch(circle)
plt.xlabel("g, real")
plt.ylabel("s, imag")
# plt.title("Phasor Plot ")
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 0.65)
plt.gca().set_aspect('equal')
# plt.legend()
plt.show()

plt.figure()
plt.imshow(np.reshape(I_sum/N_T, (H, W)), cmap = 'hot')
# plt.imshow(np.reshape((circular_mask), (H, W)))
plt.title("Original image")
# plt.colorbar()
plt.axis('off')
plt.show()

# plot the masked image 
plt.figure()
plt.imshow(masked_image_2D/N_T, cmap = 'hot')
# plt.imshow(np.reshape((circular_mask), (H, W)))
plt.title("Masked image with selected regions on phasor")
# plt.colorbar()
plt.axis('off')
plt.show()


#%% plot phasors

# # unfiltered 
# plt.figure(figsize=(6, 6))
# theta = np.linspace(0, np.pi, 1000)  
# g_semicircle = 0.5 + 0.5 * np.cos(theta)
# s_semicircle = 0.5 * np.sin(theta)
# plt.plot(g_semicircle, s_semicircle, color='black')
# # add lifetime marks on the semicircle 
# lifetimes_to_mark = [0.5, 1, 2, 3, 4, 6, 10]  # ns
# for tau in lifetimes_to_mark:
#     g = 1 / (1 + (omega * tau) ** 2)
#     s = (omega * tau) / (1 + (omega * tau) ** 2)
#     plt.scatter(g, s, color = 'black', s = 12, marker = 'o')
#     plt.text(g, s, f"{tau} ns", fontsize = 8, ha='right')   
# # color code phasor density 
# phasors = np.vstack([g_corrected, s_corrected]) # turn to 2D array 
# # perform KDE 
# kde = gaussian_kde(phasors)(phasors)
# # sort phasors by density for visualization purpose 
# idx = kde.argsort()
# g_sorted = g_corrected[idx]
# s_sorted = s_corrected[idx]
# kde_sorted = kde[idx]    
# # KDE color-coded scatter plot
# plt.scatter(g_sorted, s_sorted, c=kde_sorted, s=4, cmap='viridis', alpha=0.8, edgecolors='none')
# # plt.scatter(g_corrected, s_corrected, s=1, alpha=0.3, c='blue')
# # plt.colorbar()
# plt.xlabel("g")
# plt.ylabel("s")
# # plt.title("Phasor Plot ")
# plt.xlim(-0.1, 1.1)
# plt.ylim(-0.1, 0.65)
# plt.gca().set_aspect('equal')
# # plt.legend()
# plt.show()


# # filtered 
# plt.figure(figsize=(6, 6))
# theta = np.linspace(0, np.pi, 1000)  
# g_semicircle = 0.5 + 0.5 * np.cos(theta)
# s_semicircle = 0.5 * np.sin(theta)
# plt.plot(g_semicircle, s_semicircle, color='black')
# # add lifetime marks on the semicircle 
# lifetimes_to_mark = [0.5, 1, 2, 3, 4, 6, 10]  # ns
# for tau in lifetimes_to_mark:
#     g = 1 / (1 + (omega * tau) ** 2)
#     s = (omega * tau) / (1 + (omega * tau) ** 2)
#     plt.scatter(g, s, color = 'black', s = 12, marker = 'o')
#     plt.text(g, s, f"{tau} ns", fontsize = 8, ha='right')
# # color code phasor density 
# phasors = np.vstack([g_corrected, s_corrected]) # turn to 2D array 
# # perform KDE 
# kde = gaussian_kde(phasors)(phasors)
# # sort phasors by density for visualization purpose 
# idx = kde.argsort()
# filt_g_sorted = med_filt_g[idx]
# filt_s_sorted = med_filt_s[idx]
# kde_sorted = kde[idx]    
# # KDE color-coded scatter plot
# plt.scatter(filt_g_sorted, filt_s_sorted, c=kde_sorted, s=4, cmap='viridis', alpha=0.8, edgecolors='none')
# # plt.scatter(g_corrected, s_corrected, s=1, alpha=0.3, c='blue')
# # plt.colorbar()
# plt.xlabel("g")
# plt.ylabel("s")
# # plt.title("Phasor Plot ")
# plt.xlim(-0.1, 1.1)
# plt.ylim(-0.1, 0.65)
# plt.gca().set_aspect('equal')
# # plt.legend()
# plt.show()


#%% debug 

# # check intensity histogram 
# plt.hist(I_sum, bins=100, log=True)
# plt.axvline(intensity_threshold, color='red', label=f'Threshold = {np.round(intensity_threshold, 2)}%')
# plt.title("Intensity Histogram")
# plt.xlabel("Total intensity across gates")
# plt.ylabel("Pixel count (log)")
# plt.legend()
# plt.show()

# # intensity threshold masking 
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

