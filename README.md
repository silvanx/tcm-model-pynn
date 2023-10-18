The thalamo-cortical microcircuit (TCM) spiking neural network model developed by AmirAli Farokhniaee in Matlab, translated to PyNN by John Fleming.

The original paper: _Cortical network effects of subthalamic deep brain stimulationin a thalamo-cortical microcircuit model_ by AmirAli Farokhniaee Ph.D. and Madeleine Lowery Ph.D. 2021 Journal of Neural Engineering J. Neural Eng.18(2021) 056006 https://doi.org/10.1088/1741-2552/abee50

# Quickstart guide
## Using Conda
### Setup
0. Install NEURON from https://neuron.yale.edu/ftp/neuron/versions/v8.0/8.0.0/nrn-8.0.0.w64-mingw-py-27-35-36-37-38-39-setup.exe
1. Create a Conda environment using the attached .yml file
```
conda env create --file ./environment.yml
```
2. Activate the conda environment
```
conda activate pynn-tcm-env
```
3. Compile the NEURON mechanisms
```
nrnivmodl
```

### Running the model
To run the model use
```
python TCM_Build_Run.py
```
The simulation results will be saved to an output directory, the name of the output directory will be printed to the standard output.

To plot the results use
```
python plot_simulation_results.py ./Results/<result_dir>
```

## Using Docker
1. Build the image
```
docker build -t pynn-tcm-model .
```
2. Run the container
```
docker run pynn-tcm-model
```
3. Copy the results from the container
```
docker cp <container-name>:/usr/app/src/TCM/Results/ ./
```

# Changelog
## Edition 3, 25/09/2023
  - Removed spiking threshold variability from the neuronal populations (h.Random() was causing the code to not terminate)
  - Saving the simulation parameters at the end of simulation
## Edition 2, 19/04/2023
  - Upscaling of populations (5e2 and 1e3 order of magnitude) is ensured to work well. Above threshold bias currents are used for numerical stability.
  - non-PD (normal) couplings are introduced as well as PD couplings that was already in use
## Edition 1, 28/03/2023
