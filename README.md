The thalamo-cortical microcircuit (TCM) spiking neural network model developed by AmirAli Farokhniaee in Matlab, translated to PyNN by John Fleming.

The original paper: _Cortical network effects of subthalamic deep brain stimulationin a thalamo-cortical microcircuit model_ by AmirAli Farokhniaee Ph.D. and Madeleine Lowery Ph.D. 2021 Journal of Neural Engineering J. Neural Eng.18(2021) 056006 https://doi.org/10.1088/1741-2552/abee50

The original translation by John Fleming, editor: AmirAli Farokhniaee.

# Changelog
## Edition 3, 25/09/2023
  - Removed spiking threshold variability from the neuronal populations (h.Random() was causing the code to not terminate)
  - Saving the simulation parameters at the end of simulation
## Edition 2, 19/04/2023
  - Upscaling of populations (5e2 and 1e3 order of magnitude) is ensured to work well. Above threshold bias currents are used for numerical stability.
  - non-PD (normal) couplings are introduced as well as PD couplings that was already in use
## Edition 1, 28/03/2023