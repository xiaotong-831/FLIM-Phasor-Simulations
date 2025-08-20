# FLIM Phasor Simulations and Analysis 

This repository contains code to **simulate fluorescence decays, time gates, and how different approaches to analyze FLIM data including curve fitting, weighted lifetime, and phasor analysis affect the measurement accuracy**. 

## Single exponential decay simulation 

The 'Single_exponential_decay_simulations.py' simulates fluorescence decay characteristics of single lifetime molecular species. The code invetsigates how the three different data analysis approaches: curve fitting, weighted lifetime, phasor analysis method, affect the measurement accuracy. We also change the simulation parameters: gate width, gate delay, and the total detected photons, to study how system settings affect the acquisition speed, signal-to-noise ratio (SNR) and overall quality and performance of the measurement. 

## Multi-exponential decay phasor simulation 

The `Multi_exponential_decay_simulations` function contains code to simulate double exponential decays mixtures with different fractions of free and bound NADH. The purpose of this simulation is to (a) learn how differeny system parameters affect the phasor results and to (b) find optimal parameters to balance the trade-off between acquisition speed, lifetime resolution/separability and photon flux. 

## Phasor analysis simulation 

The 'Phasor analysis for experimental data.py' contains code for uses to import their FLIM calibration and experimental data and plot phasors to perform analysis. The code includes calibration procedure, phasor analysis, with the degree of freedom to add or modify intensity threshold masks and filters, and the function to select regions on the phasor plots using circular masks and map the corresponding pixels on the original image. This function enables the user to select the region of interest on the phasor plots and locate contributing pixels in order to identify the source of fluorescence, noisy phasors and so on.





