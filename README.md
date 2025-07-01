# FLIM Phasor Simulations and Analysis 

This repository contains code to **simulate fluorescence decays, time gates, and how different approaches to analyze FLIM data including curve fitting, weighted lifetime, and phasor analysis affect the measurement accuracy**. 

## 🧪 Simulation Code (Synthetic FLIM Image Generation)

The `single` function simulates synthetic FLIM data with spatially varying decay characteristics. It creates both a synthetic signal image and a reference image for calibration or validation. The synthetic signal image, aiming to simulate biological structures generates randomly blobs with different decay times (biologically corresponding to different species for example). Phasor-analysis allows the visualization of the different regions. To imitate experimental conditions, photon numbers follow a Poisson distribution. The expected numbers of photons follow a bi-exponential (supposing that we have two contributions, from free and bound NADH for example) with randomized ratios for the contributions. The ratios follow a uniform distribution with a chosen mean (min, max = mean +/- 0.2).





# FLIM Synthetic Data Generator and Phasor Analysis

This repository contains code to **simulate synthetic Fluorescence Lifetime Imaging Microscopy (FLIM) data** and perform **filtering (median, wavelet) and visualization** using [**phasorpy**](https://www.phasorpy.org/docs/dev/release/#id1) and [**pawflim**](https://github.com/maurosilber/pawflim).

---

## 📦 Dependencies

Before using this code, make sure you have the following packages installed:

- [`phasorpy==0.5`](https://www.phasorpy.org/docs/dev/release/#id1): Phasor analysis library
- [`pawflim`](https://github.com/maurosilber/pawflim): Tools for advanced phasor-based filtering
