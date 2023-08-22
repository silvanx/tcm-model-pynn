""" -*- coding: utf-8 -*-

Description: The thalamo-cortical microcircuit (TCM) spiking neural netwrok model developed by AmirAli Farokhniaee in Matlab, trnaslated to PyNN by John Flemming.
             The original paper: Cortical network effects of subthalamic deep brain stimulationin a thalamo-cortical microcircuit model
                                 by AmirAli Farokhniaee Ph.D. and Madeleine Lowery Ph.D. 2021 Journal of Neural Engineering 
                                 J. Neural Eng.18(2021) 056006 https://doi.org/10.1088/1741-2552/abee50
             The original translatoin by John Fleming, eidtor: AmirAli Farokhniaee.

# Edition 1, 28/03/2023
# Edition 2, 19/04/2023 - Upscaling of populations (5e2 and 1e3 order of magnitude) is ensured to work well. Above threshold bias currents are used for numerical stability.
						  non-PD (normal) couplings are introduced as well as PD couplings that was already in use
"""

import neuron
h = neuron.h


from math import pi
from pyNN.neuron import setup, run, Population, Projection, ArrayConnector, OneToOneConnector, AllToAllConnector, TsodyksMarkramSynapse, DCSource, NoisyCurrentSource, initialize, SpikeSourcePoisson, StaticSynapse, FixedProbabilityConnector, reset
from pyNN.random import RandomDistribution, NumpyRNG
from pyNN import space
from Cortical_Basal_Ganglia_Cell_Classes import Izhikevich_Type
import numpy as np
#import math
#import scipy.signal as signal
#import os
import sys
from timeit import default_timer as timer
#import time
#import matplotlib.pyplot as plt
#from neo.io import NixIO
#from pyNN.utility.plotting import Figure, Panel

# Import global variables for generating the streams for the membrane noise
import Global_Variables as GV

# Load ranstream hoc template for generating random streams
h.load_file("ranstream.hoc")


"""	Tsodyks-Markram Synapse Parameters - Taken from AmirAli's Matlab model: see the supplementary material!
------------------------------------------------------------------------------------------------------
		Synapse Type			|  U	| tau_rec	|	tau_facil	|	 A	    |	tau_s 	|  delay | 	
------------------------------------------------------------------------------------------------------
	Excitatory Facilitating		| 0.09	|  	138	    |  	   670		|   0.2  	|	 3  	|	 1	 |	
------------------------------------------------------------------------------------------------------
	Excitatory Depressing		| 0.5	|  	671    |  	   17		|   0.63  	|	 3	    |	 1	 |
------------------------------------------------------------------------------------------------------
	Excitatory Pseudolinear		| 0.29	|  	329    |  	   326		|   0.17  	|	 3	    |	 1	 |	
------------------------------------------------------------------------------------------------------
	Inhibitory Facilitating		| 0.016 |  	45	    |  	   376		|   0.08  	|	 11     |    1   |
------------------------------------------------------------------------------------------------------
	Inhibitory Depressing		| 0.25	|  	706	    |  	   21		|   0.75  	|	 11	    |	 1	 |	
------------------------------------------------------------------------------------------------------
	Inhibitory Pseudolinear		| 0.32	|  	144    |  	   62		|   0.17 	|	 11 	|	 1	 |			
------------------------------------------------------------------------------------------------------
"""
""" Neuron Parameters - Taken from AmirAli's Matlab model for subthreshold regime of TCM: see the supplementary material!
---------------------------------------------------------------------------------------------------------
		Neuron Type					|	a	|	b	|	c	|	d	|	v_thresh 	|	Bias current	|  	
---------------------------------------------------------------------------------------------------------
	Regular Spiking (RS)			| 0.02	|  0.2	| -65.0	|  8.0	|	 30.0  		|	 	3.5697 		|	
---------------------------------------------------------------------------------------------------------
	Intrinsically Bursting (IB)		| 0.02	|  	0.2	| -55.0 |  4.0	|	 30.0	    |	 	3.6689	 	| 
----------------------------------------------------------------------------------------------------------
	Fast Spiking (FS)				| 0.1	|  0.2  | -65.0	|  2.0  |	 30.0	    |	 	3.8672	 	|	
----------------------------------------------------------------------------------------------------------
	Low Threshold Spiking (LTS)		| 0.02 	|  0.25 | -65.0	|  2.0  |	 30.0     	|    	0.4958   	|
----------------------------------------------------------------------------------------------------------
	Thalamocortical Relay (TC)		| 0.02	|  0.25 | -65.0	|  0.05 |	 30.0	    |	 	0.6941	 	|	
----------------------------------------------------------------------------------------------------------
	Thalamic Reticular (TR)			| 0.02	|  0.25 | -65.0 |  2.05 |	 30.0 	    |	 	0.6941	 	|			
----------------------------------------------------------------------------------------------------------	
"""
"""Maximum coupling weights from row to columns of TCM substructures
- Two sets of weights are given for noraml and PD conditions; see the last supplementary figure! 
----------------------------------------------------------------------------------------------------------
NORMAL;
-------
		|	S	|	M	|	D	|	CI	|	TCR	|	TRN	
		------------------------------------------------
	S	|  -10	|  	300	|	300	|  200	|	0	|	0	
	M	|   10	|  -10	|	0	|  200	|	0	|	0
	D	|  500	|	0	|  -10	|  200	|  700	|  700
	CI	| -500	|  -300	| -7500	| -500	|	0	|   0
	TCR	|	0	|	0	|	10	|  10	|	0	|  1000
	TRN	|	0	|	0	|	0	|	0	| -500	|  -50

PD;
----
		|	S	|	M	|	D	|	CI	|	TCR	|	TRN	
		------------------------------------------------
	S	|  -50	|   10	|  300	|  200	|	0	|	0
	M	|  300	|  -50	|	0	|  200	|	0	|	0
	D	|  500	|	0	|  -50	|  200	|  100	|  100
	CI	| -750	| -750	| -5000	|  -50	|	0	|	0
	TCR	|	0	|	0	|  1000	|  1000	|	0	|  500
	TRN	|	0	|	0	|	0	|	0	| -2500	|  -50
----------------------------------------------------------------------------------------------------------
"""

if __name__ == '__main__':
    
    start = timer()
    #initialize simulation
    simulation_timestep = 0.1
    setup(timestep=simulation_timestep, rngseed=3695456) 
    simulation_time = 10000.0		# ms
    pop_size = 100
    PD_factor = 800

    print("\nSetting simulation time to",simulation_time/1000,"(s) with time step =",simulation_timestep,"(ms)")

    # Set the global offset variable for the random streams
    GV.random_stream_offset = int(simulation_time * (1.0/simulation_timestep))


    # Use CVode to calculate i_membrane_ for fast LFP calculation
    cvode = h.CVode() #cvode is a variable time step methods
    cvode.active(0)

    # Create random distribution for initial cell membrane voltages
    v_init = RandomDistribution('uniform',(-70.0, -60.0))
    parameter_variability = RandomDistribution('uniform',(0, 1))

    # Population Size
    #pop_size = 100
    print("\nNeuronal population size, order of magnitude", pop_size)

    # Variation of TRN noise
    #noise_rescale_factor = float(sys.argv[2])
    noise_rescale_factor = 1


    # Neuron (and Exponential Synapse) Parameters - Taken from AmirAli's Matlab model: see the supplementary material!
    RS_parameters =  {'a_': 0.02, 'b': 0.2,  'c': -65.0, 'd': 8.0,  'thresh': 30.0, 'thresh_noise': 0.5, 'bias_current_amp': 4, 'noise_mean': 0, 'noise_stdev': 1.5, 'e_rev_e': 0.0, 'tau_e': 3.0, 'e_rev_i': -70.0, 'tau_i': 11.0}
    IB_parameters =  {'a_': 0.02, 'b': 0.2,  'c': -55.0, 'd': 4.0,  'thresh': 30.0, 'thresh_noise': 0.5, 'bias_current_amp': 4, 'noise_mean': 0, 'noise_stdev': 1.5, 'e_rev_e': 0.0, 'tau_e': 3.0, 'e_rev_i': -70.0, 'tau_i': 11.0}
    FS_parameters =  {'a_': 0.1,  'b': 0.2,  'c': -65.0, 'd': 2.0,  'thresh': 30.0, 'thresh_noise': 0.5, 'bias_current_amp': 4, 'noise_mean': 0, 'noise_stdev': 1.5, 'e_rev_e': 0.0, 'tau_e': 3.0, 'e_rev_i': -70.0, 'tau_i': 11.0}
    LTS_parameters = {'a_': 0.02, 'b': 0.25, 'c': -65.0, 'd': 2.0,  'thresh': 30.0, 'thresh_noise': 0.5, 'bias_current_amp': 1, 'noise_mean': 0, 'noise_stdev': 1.5,  'e_rev_e': 0.0, 'tau_e': 3.0, 'e_rev_i': -70.0, 'tau_i': 11.0}
    Rel_TC_parameters =  {'a_': 0.02, 'b': 0.25, 'c': -65.0, 'd': 0.05, 'thresh': 30.0, 'thresh_noise': 0.5, 'bias_current_amp': 1, 'noise_mean': 0, 'noise_stdev': 1.5, 'e_rev_e': 0.0, 'tau_e': 3.0, 'e_rev_i': -70.0, 'tau_i': 11.0}
    Ret_parameters =  {'a_': 0.02, 'b': 0.25, 'c': -65.0, 'd': 2.05, 'thresh': 30.0, 'thresh_noise': 0.5, 'bias_current_amp': 1, 'noise_mean': 0, 'noise_stdev': noise_rescale_factor * 1.5, 'e_rev_e': 0.0, 'tau_e': 3.0, 'e_rev_i': -70.0, 'tau_i': 11.0}

    ###----------------------------------------- Neuron Populations ---------------------------------------------------###
    print("\nAssembling neuronal populatoins within cortical layers and thalamic nuclei...")
    # Cortical layers and Interneuron Assemblies:
    ###--------------------------------------------------- S Layer ----------------------------------------------------### 
    # S layer - Regular Spiking Neurons:
    S_layer_RS_neuron_num = round(0.5*pop_size)
    S_layer_RS_Population = Population(S_layer_RS_neuron_num, Izhikevich_Type(**RS_parameters))

    # Set the seed for the random object for parameter variation
    np.random.seed(3695) # when we set the seed, the random number generation will be deterministic

    # Add variability to c values
    r = np.random.uniform(0, 1, S_layer_RS_neuron_num)
    c_value = S_layer_RS_Population.get('c')
    updated_c_values = c_value + 15*(r**2)
    S_layer_RS_Population.set(c = updated_c_values)

    # Add variability to d values
    d_value = S_layer_RS_Population.get('d')
    updated_d_values = d_value - 0.6*(r**2)
    S_layer_RS_Population.set(d = updated_d_values)

    # S layer - Intermittent Bursting Neurons
    S_layer_IB_neuron_num = pop_size - S_layer_RS_neuron_num
    S_layer_IB_Population = Population(S_layer_IB_neuron_num, Izhikevich_Type(**IB_parameters))

    # Add variability to c values
    r = np.random.uniform(0, 1, S_layer_IB_neuron_num)
    c_value = S_layer_IB_Population.get('c')
    updated_c_values = c_value + 15*(r**2)
    S_layer_IB_Population.set(c = updated_c_values)

    # Add variability to d values
    d_value = S_layer_IB_Population.get('d')
    updated_d_values = d_value - 0.6*(r**2)
    S_layer_IB_Population.set(d = updated_d_values)

    # Make assembly for the S layer - Contains RS and IB neurons
    S_layer_neurons = S_layer_RS_Population + S_layer_IB_Population

    ###--------------------------------------------------- M Layer ----------------------------------------------------### 
    # M layer - Regular Spiking Neurons:
    M_layer_RS_neuron_num = pop_size
    M_layer_RS_Population = Population(M_layer_RS_neuron_num, Izhikevich_Type(**RS_parameters))

    # Add variability to c values
    r = np.random.uniform(0, 1, M_layer_RS_neuron_num)
    c_value = M_layer_RS_Population.get('c')
    updated_c_values = c_value + 15*(r**2)
    M_layer_RS_Population.set(c = updated_c_values)

    # Add variability to d values
    d_value = M_layer_RS_Population.get('d')
    updated_d_values = d_value - 0.6*(r**2)
    M_layer_RS_Population.set(d = updated_d_values)

    # Make assembly for the M layer - Contains only RS neurons
    M_layer_neurons = M_layer_RS_Population

    ###--------------------------------------------------- D Layer ----------------------------------------------------### 
    # D layer - Regular Spiking Neurons:
    D_layer_RS_neuron_num = round(0.7*pop_size)
    D_layer_RS_Population = Population(D_layer_RS_neuron_num, Izhikevich_Type(**RS_parameters))

    # Add variability to c values
    r = np.random.uniform(0, 1, D_layer_RS_neuron_num)
    c_value = D_layer_RS_Population.get('c')
    updated_c_values = c_value + 15*(r**2)
    D_layer_RS_Population.set(c = updated_c_values)

    # Add variability to d values
    d_value = D_layer_RS_Population.get('d')
    updated_d_values = d_value - 0.6*(r**2)
    D_layer_RS_Population.set(d = updated_d_values)

    # D layer - Intermittent Bursting Neurons
    D_layer_IB_neuron_num = pop_size - D_layer_RS_neuron_num
    D_layer_IB_Population = Population(D_layer_IB_neuron_num, Izhikevich_Type(**IB_parameters))

    # Add variability to c values
    r = np.random.uniform(0, 1, D_layer_IB_neuron_num)
    c_value = D_layer_IB_Population.get('c')
    updated_c_values = c_value + 15*(r**2)
    D_layer_IB_Population.set(c = updated_c_values)

    # Add variability to d values
    d_value = D_layer_IB_Population.get('d')
    updated_d_values = d_value - 0.6*(r**2)
    D_layer_IB_Population.set(d = updated_d_values)

    # Make assembly for the D layer - Contains RS and IB neurons
    D_layer_neurons = D_layer_RS_Population + D_layer_IB_Population

    ###--------------------------------------------------- Cortical Interneurons ----------------------------------------------------### 
    # Cortical Interneurons (CI) - Fast Spiking Interneurons:
    CI_FS_neuron_num = round(0.5*pop_size)
    CI_FS_Population = Population(CI_FS_neuron_num, Izhikevich_Type(**FS_parameters))

    # Add variability to a values
    r = np.random.uniform(0, 1, CI_FS_neuron_num)
    a_value = CI_FS_Population.get('a_')
    updated_a_values = a_value + 0.008*r
    CI_FS_Population.set(a_ = updated_a_values)

    # Add variability to b values
    b_value = CI_FS_Population.get('b')
    updated_b_values = b_value - 0.005*r
    CI_FS_Population.set(b = updated_b_values)

    # Cortical Interneurons (CI) - Low-Threshold Spiking Interneurons:
    CI_LTS_neuron_num = pop_size - CI_FS_neuron_num
    CI_LTS_Population = Population(CI_LTS_neuron_num, Izhikevich_Type(**LTS_parameters))

    # Add variability to a values
    r = np.random.uniform(0, 1, CI_LTS_neuron_num)
    a_value = CI_LTS_Population.get('a_')
    updated_a_values = a_value + 0.008*r
    CI_LTS_Population.set(a_ = updated_a_values)

    # Add variability to b values
    b_value = CI_LTS_Population.get('b')
    updated_b_values = b_value - 0.005*r
    CI_LTS_Population.set(b = updated_b_values)

    # Make assembly for the cortical interneurons - Contains both Fast Spiking and Low-Threshold Spiking Interneurons
    CI_neurons = CI_FS_Population + CI_LTS_Population

    ###--------------------------------------------------- Thalamic Reticular Nucleus (TRN) ----------------------------------------------------### 
    TRN_TR_neuron_num = round(0.4*pop_size)
    TRN_TR_Population = Population(TRN_TR_neuron_num, Izhikevich_Type(**Ret_parameters))

    # Add variability to a values
    r = np.random.uniform(0, 1, TRN_TR_neuron_num)
    a_value = TRN_TR_Population.get('a_')
    updated_a_values = a_value + 0.008*r
    TRN_TR_Population.set(a_ = updated_a_values)

    # Add variability to b values
    b_value = TRN_TR_Population.get('b')
    updated_b_values = b_value - 0.005*r
    TRN_TR_Population.set(b = updated_b_values)	

    # Make assembly for Thalamic Reticular Nucleus
    TRN_TR_neurons = TRN_TR_Population

    ###--------------------------------------------------- Thalamo-Cortical Relay Nucleus (TCR) ----------------------------------------------------### 
    TCR_TC_neuron_num = pop_size
    TCR_TC_Population = Population(TCR_TC_neuron_num, Izhikevich_Type(**Rel_TC_parameters))

    # Add variability to a values
    r = np.random.uniform(0, 1, TCR_TC_neuron_num)
    a_value = TCR_TC_Population.get('a_')
    updated_a_values = a_value + 0.008*r
    TCR_TC_Population.set(a_ = updated_a_values)

    # Add variability to b values
    b_value = TCR_TC_Population.get('b')
    updated_b_values = b_value - 0.005*r
    TCR_TC_Population.set(b = updated_b_values)

    # Make assembly for Thalamo-Cortical Relay Nucleus
    TCR_TC_neurons = TCR_TC_Population

    # Make assembly of all cells in the thalamo-cortical network
    all_TC_neurons = S_layer_neurons + M_layer_neurons + D_layer_neurons + CI_neurons + TRN_TR_neurons + TCR_TC_neurons

    ###----------------------------------------- Define Connections ---------------------------------------------------###
    # Set up coupling within the network
    print("\nSetting up connections:")
    # Generate random numbers from gaussian distribution for setting up synaptic weights as in Matlab model
    rng = NumpyRNG(seed=8658764) 
    weight_uniform_dist = RandomDistribution('uniform', [0, 1],rng=rng)

    r_s = weight_uniform_dist.next(S_layer_neurons.size)
    r_m = weight_uniform_dist.next(M_layer_neurons.size)
    r_d = weight_uniform_dist.next(D_layer_neurons.size)
    r_ins = weight_uniform_dist.next(CI_neurons.size)
    r_ret =weight_uniform_dist.next(TRN_TR_neurons.size)
    r_rel = weight_uniform_dist.next(TCR_TC_neurons.size)

    # All-to-All Connectivity Structure:
    connectivity_structure = np.ones((pop_size,pop_size),np.int_)
    print("Defining coupling distribution; all to all uniformly distributed random coupling.\nSetting up the type of synapses and their distributions within and between each layer and nuclei...")

    # Define the adjacency matrices for each population (each connection is rescaled by max value below in projection definitions)
    r_s = r_s/S_layer_neurons.size #FK ? 
    r_m = r_m/M_layer_neurons.size
    r_d = r_d/D_layer_neurons.size
    r_ins = r_ins/CI_neurons.size
    r_ret = r_ret/TRN_TR_neurons.size
    r_rel = r_rel/TCR_TC_neurons.size

    # TM Synapse Parameters:
    # Percentage of synaptic weight values which correspond to each TM synapse type (Facilitating, Depressing and Pseudolinear)
    Exc_Fac_weight = 0.2					# 20 % of excitatory syanpses are facilitating
    Exc_Dep_weight = 0.63					# 63 % of excitatory synapses are depressing 
    Exc_Pseudo_weight = 0.17				# 17 % of excitatory synapses are pseudolinear
    Inh_Fac_weight = 0.08					# 8 % of inhibitory syanpses are facilitating
    Inh_Dep_weight = 0.75					# 75 % of inhibitory synapses are depressing 
    Inh_Pseudo_weight = 0.17				# 17 % of inhibitory synapses are pseudolinear

    # Excitatory synaptic dynamics parameters
    Uef = 0.09
    Ued = 0.05
    Uep = 0.29
    tau_rec_ef = 138
    tau_rec_ed = 671
    tau_rec_ep = 329
    tau_facil_ef = 670
    tau_facil_ed = 17
    tau_facil_ep = 326

    # Inhibitory synaptic dynamics parameters
    Uif = 0.016
    Uid = 0.25
    Uip = 0.32
    tau_rec_if = 45
    tau_rec_id = 706
    tau_rec_ip = 144
    tau_facil_if = 376
    tau_facil_id = 211
    tau_facil_ip = 62

    #The synaptic ratios that will be multiplied in the adjacancy matrix calculated based on F synapses:
    Exc_syn_ratio_D = Exc_Dep_weight/Exc_Fac_weight
    Exc_syn_ratio_P = Exc_Pseudo_weight/Exc_Fac_weight
    Inh_syn_ratio_D = Inh_Dep_weight/Inh_Fac_weight
    Inh_syn_ratio_P = Inh_Pseudo_weight/Inh_Fac_weight

    """ ---------------------------------------------Synaptic weight values: ---------------------------------------------"""
    #PD_factor = 2.5							# Scaling factor for simulating normal conditions
    #PD_factor = 800							# Scaling factor for simulating parkinsonian conditions (Maybe ~8*pop_size)
    print("Factor: ", PD_factor)
    print("Be patient, it takes a while...")
    # Define scaling factor for rescaling original publication synaptic weights for NEURON - more details of current rescaling are described in Izh.mod
    cell_diam = 10.0/pi
    cell_L = 10.0
    cell_cm = 1.0
    synaptic_rescale_factor= pi * cell_diam * cell_L * cell_cm * 1e-5 				

    # Rescale factor for debugging the synaptic weights	- originally implemented as part of a parameter sweep to find a suitable value to match Matlab model behaviour
    #conversion_factor = float(sys.argv[2])
    #conversion_factor = 0.006024489795918
    conversion_factor = 1
    synaptic_rescale_factor = conversion_factor*synaptic_rescale_factor

    # Max Coupling Strengths within each structure
    g_S_Layer_S_Layer = synaptic_rescale_factor * 5e1/PD_factor
    g_M_Layer_M_Layer = synaptic_rescale_factor * 5e1/PD_factor
    #coupling_rescale_factor = float(sys.argv[2])
    g_D_Layer_D_Layer = 1*synaptic_rescale_factor * 5e1/PD_factor
    #g_D_Layer_D_Layer = synaptic_rescale_factor * 5e1/PD_factor
    g_CI_CI = 1.0 * synaptic_rescale_factor * 5e1/PD_factor
    #coupling_rescale_factor = float(sys.argv[2])
    g_TRN_TRN = 1.0 * synaptic_rescale_factor * 5e1/PD_factor
    g_TCR_TCR = synaptic_rescale_factor * 0/PD_factor

    # Max Coupling Strengths between structures
    # Couplings to S Layer
    g_M_Layer_S_Layer = synaptic_rescale_factor * 3e2/PD_factor
    g_D_Layer_S_Layer = synaptic_rescale_factor * 5e2/PD_factor
    g_CI_S_Layer = synaptic_rescale_factor * 7.5e2/PD_factor
    g_TRN_S_Layer = synaptic_rescale_factor * 0/PD_factor
    g_TCR_S_Layer = synaptic_rescale_factor * 0/PD_factor

    # Couplings to M Layer
    g_S_Layer_M_Layer = synaptic_rescale_factor * 1e1/PD_factor
    g_D_Layer_M_Layer = synaptic_rescale_factor * 0/PD_factor
    g_CI_M_Layer = synaptic_rescale_factor * 7.5e2/PD_factor
    g_TRN_M_Layer = synaptic_rescale_factor * 0/PD_factor
    g_TCR_M_Layer = synaptic_rescale_factor * 0/PD_factor

    # Couplings to D Layer
    g_S_Layer_D_Layer = synaptic_rescale_factor * 3e2/PD_factor
    g_M_Layer_D_Layer = synaptic_rescale_factor * 0/PD_factor
    g_CI_D_Layer = synaptic_rescale_factor * 5e3/PD_factor
    g_TRN_D_Layer = synaptic_rescale_factor * 0/PD_factor
    g_TCR_D_Layer = 1.0 * synaptic_rescale_factor * 1e3/PD_factor

    # Couplings to CI neurons
    g_S_Layer_CI = synaptic_rescale_factor * 2e2/PD_factor
    g_M_Layer_CI = synaptic_rescale_factor * 2e2/PD_factor
    g_D_Layer_CI = synaptic_rescale_factor * 2e2/PD_factor
    g_TRN_CI = synaptic_rescale_factor * 0/PD_factor
    g_TCR_CI = synaptic_rescale_factor * 1e3/PD_factor

    # Couplings to TRN neurons
    g_S_Layer_TRN = synaptic_rescale_factor * 0/PD_factor
    g_M_Layer_TRN = synaptic_rescale_factor * 0/PD_factor
    g_D_Layer_TRN = synaptic_rescale_factor * 1e2/PD_factor
    g_CI_TRN = synaptic_rescale_factor * 0/PD_factor
    g_TCR_TRN = synaptic_rescale_factor * 5e2/PD_factor

    # Couplings to TCR neurons
    g_S_Layer_TCR = synaptic_rescale_factor * 0/PD_factor
    g_M_Layer_TCR = synaptic_rescale_factor * 0/PD_factor
    g_D_Layer_TCR = synaptic_rescale_factor * 1e2/PD_factor
    g_CI_TCR = synaptic_rescale_factor * 0/PD_factor
    g_TRN_TCR = synaptic_rescale_factor * 2.5e3/PD_factor

    # Synaptic Delays
    t_d_l = 8		# Time delay between the layers in cortex and nuclei in thalamus (ms)
    t_d_wl = 1		# Time delay within a structure (ms)
    t_d_TC = 15		# Time delay between the thalamus and cortex (ms)
    t_d_CT = 20		# Time delay between the cortex and thalamus (ms)
    t_d_syn = 1		# Synaptic transmission delay (ms - fixed for all synapses in the TCM)

    ###-------------------------------------------- Coupling within populations - projections --------------------------------------------###
    # S Layer:
    # 	- Facilitating:
    S_layer_S_layer_F_proj = Projection(S_layer_neurons, S_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uif, tau_rec=tau_rec_if, tau_facil=tau_facil_if,weight=Inh_Fac_weight*g_S_Layer_S_Layer*r_s*connectivity_structure, delay=t_d_wl+t_d_syn),
                                                receptor_type='isyn')		
    # 	- Depressing:
    S_layer_S_layer_D_proj = Projection(S_layer_neurons, S_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uid, tau_rec=tau_rec_id, tau_facil=tau_facil_id,weight=Inh_Dep_weight*g_S_Layer_S_Layer*r_s*connectivity_structure, delay=t_d_wl+t_d_syn),
                                                receptor_type='isyn')	
    # 	- Pseudolinear:
    S_layer_S_layer_P_proj = Projection(S_layer_neurons, S_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uip, tau_rec=tau_rec_ip, tau_facil=tau_facil_ip,weight=Inh_Pseudo_weight*g_S_Layer_S_Layer*r_s*connectivity_structure, delay=t_d_wl+t_d_syn),
                                                receptor_type='isyn')			

    # M Layer:
    # 	- Facilitating:
    M_layer_M_layer_F_proj = Projection(M_layer_neurons, M_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uif, tau_rec=tau_rec_if, tau_facil=tau_facil_if,weight=Inh_Fac_weight*g_M_Layer_M_Layer*r_m*connectivity_structure, delay=t_d_wl+t_d_syn),
                                                receptor_type='isyn')		
    # 	- Depressing:
    M_layer_M_layer_D_proj = Projection(M_layer_neurons, M_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uid, tau_rec=tau_rec_id, tau_facil=tau_facil_id,weight=Inh_Dep_weight*g_M_Layer_M_Layer*r_m*connectivity_structure, delay=t_d_wl+t_d_syn),
                                                receptor_type='isyn')	
    # 	- Pseudolinear:
    M_layer_M_layer_P_proj = Projection(M_layer_neurons, M_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uip, tau_rec=tau_rec_ip, tau_facil=tau_facil_ip,weight=Inh_Pseudo_weight*g_M_Layer_M_Layer*r_m*connectivity_structure, delay=t_d_wl+t_d_syn),
                                                receptor_type='isyn')	

    # D Layer:
    # 	- Facilitating:
    D_layer_D_layer_F_proj = Projection(D_layer_neurons, D_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uif, tau_rec=tau_rec_if, tau_facil=tau_facil_if,weight=Inh_Fac_weight*g_D_Layer_D_Layer*r_d*connectivity_structure, delay=t_d_wl+t_d_syn),
                                                receptor_type='isyn')		
    # 	- Depressing:
    D_layer_D_layer_D_proj = Projection(D_layer_neurons, D_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uid, tau_rec=tau_rec_id, tau_facil=tau_facil_id,weight=Inh_Dep_weight*g_D_Layer_D_Layer*r_d*connectivity_structure, delay=t_d_wl+t_d_syn),
                                                receptor_type='isyn')	
    # 	- Pseudolinear:
    D_layer_D_layer_P_proj = Projection(D_layer_neurons, D_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uip, tau_rec=tau_rec_ip, tau_facil=tau_facil_ip,weight=Inh_Pseudo_weight*g_D_Layer_D_Layer*r_d*connectivity_structure, delay=t_d_wl+t_d_syn),
                                                receptor_type='isyn')

    # CI Neurons:
    # 	- Facilitating:
    CI_CI_F_proj = Projection(CI_neurons, CI_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uif, tau_rec=tau_rec_if, tau_facil=tau_facil_if,weight=Inh_Fac_weight*g_CI_CI*r_ins*connectivity_structure, delay=t_d_wl+t_d_syn),
                                                receptor_type='isyn')	

    # 	- Depressing:
    CI_CI_D_proj = Projection(CI_neurons, CI_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uid, tau_rec=tau_rec_id, tau_facil=tau_facil_id,weight=Inh_Dep_weight*g_CI_CI*r_ins*connectivity_structure, delay=t_d_wl+t_d_syn),
                                                receptor_type='isyn')	

    # 	- Pseudolinear:
    CI_CI_P_proj = Projection(CI_neurons, CI_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uip, tau_rec=tau_rec_ip, tau_facil=tau_facil_ip,weight=Inh_Pseudo_weight*g_CI_CI*r_ins*connectivity_structure, delay=t_d_wl+t_d_syn),
                                                receptor_type='isyn')	

    # TRN Neurons:
    # 	- Facilitating:
    TRN_TRN_F_proj = Projection(TRN_TR_neurons, TRN_TR_neurons,
                                                #connector=ArrayConnector(connectivity_structure[0:TRN_TR_neurons.size,0:TRN_TR_neurons.size]),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uif, tau_rec=tau_rec_if, tau_facil=tau_facil_if,weight=Inh_Fac_weight*g_TRN_TRN*r_ret*connectivity_structure[0:TRN_TR_neurons.size,0:TRN_TR_neurons.size], delay=t_d_wl+t_d_syn),
                                                receptor_type='isyn')	

    # 	- Depressing:
    TRN_TRN_D_proj = Projection(TRN_TR_neurons, TRN_TR_neurons,
                                                #connector=ArrayConnector(connectivity_structure[0:TRN_TR_neurons.size,0:TRN_TR_neurons.size]),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uid, tau_rec=tau_rec_id, tau_facil=tau_facil_id,weight=Inh_Dep_weight*g_TRN_TRN*r_ret*connectivity_structure[0:TRN_TR_neurons.size,0:TRN_TR_neurons.size], delay=t_d_wl+t_d_syn),
                                                receptor_type='isyn')	

    # 	- Pseudolinear:
    TRN_TRN_P_proj = Projection(TRN_TR_neurons, TRN_TR_neurons,
                                                #connector=ArrayConnector(connectivity_structure[0:TRN_TR_neurons.size,0:TRN_TR_neurons.size]),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uip, tau_rec=tau_rec_ip, tau_facil=tau_facil_ip,weight=Inh_Pseudo_weight*g_TRN_TRN*r_ret*connectivity_structure[0:TRN_TR_neurons.size,0:TRN_TR_neurons.size], delay=t_d_wl+t_d_syn),
                                                receptor_type='isyn')	

    ###-------------------------------------------- Coupling between populations - projections --------------------------------------------###
    ## S layer:
    #	1)  M layer -> S layer - 
    # 	- Facilitating:
    M_layer_S_layer_F_proj = Projection(M_layer_neurons, S_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uef, tau_rec=tau_rec_ef, tau_facil=tau_facil_ef,weight=Exc_Fac_weight*g_M_Layer_S_Layer*r_s*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='esyn')

    # 	- Depressing:
    M_layer_S_layer_D_proj = Projection(M_layer_neurons, S_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Ued, tau_rec=tau_rec_ed, tau_facil=tau_facil_ed,weight=Exc_Dep_weight*g_M_Layer_S_Layer*r_s*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='esyn')

    # 	- Pseudolinear:
    M_layer_S_layer_P_proj = Projection(M_layer_neurons, S_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uep, tau_rec=tau_rec_ep, tau_facil=tau_facil_ep,weight=Exc_Pseudo_weight*g_M_Layer_S_Layer*r_s*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='esyn')

    #	2)  D layer -> S layer - 
    # 	- Facilitating:											
    D_layer_S_layer_F_proj = Projection(D_layer_neurons, S_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uef, tau_rec=tau_rec_ef, tau_facil=tau_facil_ef,weight=Exc_Fac_weight*g_D_Layer_S_Layer*r_s*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='esyn')	

    # 	- Depressing:											
    D_layer_S_layer_D_proj = Projection(D_layer_neurons, S_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Ued, tau_rec=tau_rec_ed, tau_facil=tau_facil_ed,weight=Exc_Dep_weight*g_D_Layer_S_Layer*r_s*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='esyn')	

    # 	- Pseudolinear:											
    D_layer_S_layer_P_proj = Projection(D_layer_neurons, S_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uep, tau_rec=tau_rec_ep, tau_facil=tau_facil_ep,weight=Exc_Pseudo_weight*g_D_Layer_S_Layer*r_s*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='esyn')

    #	3)  CI neurons -> S layer - 	
    # 	- Facilitating:	
    CI_S_layer_F_proj 	   = Projection(CI_neurons, S_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uif, tau_rec=tau_rec_if, tau_facil=tau_facil_if,weight=Inh_Fac_weight*g_CI_S_Layer*r_s*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='isyn')

    # 	- Depressing:	
    CI_S_layer_D_proj 	   = Projection(CI_neurons, S_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uid, tau_rec=tau_rec_id, tau_facil=tau_facil_id,weight=Inh_Dep_weight*g_CI_S_Layer*r_s*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='isyn')

    # 	- Pseudolinear:	
    CI_S_layer_P_proj 	   = Projection(CI_neurons, S_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uip, tau_rec=tau_rec_ip, tau_facil=tau_facil_ip,weight=Inh_Pseudo_weight*g_CI_S_Layer*r_s*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='isyn')

    ## M layer:
    #	1)  S layer -> M layer - 
    # 	- Facilitating:
    S_layer_M_layer_F_proj = Projection(S_layer_neurons, M_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uef, tau_rec=tau_rec_ef, tau_facil=tau_facil_ef,weight=Exc_Fac_weight*g_S_Layer_M_Layer*r_m*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='esyn')	

    # 	- Depressing:
    S_layer_M_layer_D_proj = Projection(S_layer_neurons, M_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Ued, tau_rec=tau_rec_ed, tau_facil=tau_facil_ed,weight=Exc_Dep_weight*g_S_Layer_M_Layer*r_m*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='esyn')

    # 	- Pseudolinear:
    S_layer_M_layer_P_proj = Projection(S_layer_neurons, M_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uep, tau_rec=tau_rec_ep, tau_facil=tau_facil_ep,weight=Exc_Pseudo_weight*g_S_Layer_M_Layer*r_m*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='esyn')

    #	2)  CI Neurons -> M layer - 
    # 	- Facilitating:
    CI_M_layer_F_proj = Projection(CI_neurons, M_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uif, tau_rec=tau_rec_if, tau_facil=tau_facil_if,weight=Inh_Fac_weight*g_CI_M_Layer*r_m*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='isyn')

    # 	- Depressing:
    CI_M_layer_D_proj = Projection(CI_neurons, M_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uid, tau_rec=tau_rec_id, tau_facil=tau_facil_id,weight=Inh_Dep_weight*g_CI_M_Layer*r_m*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='isyn')

    # 	- Pseudolinear:
    CI_M_layer_P_proj = Projection(CI_neurons, M_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uip, tau_rec=tau_rec_ip, tau_facil=tau_facil_ip,weight=Inh_Pseudo_weight*g_CI_M_Layer*r_m*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='isyn')

    ## D layer:
    #	1)  S layer -> D layer - 
    # 	- Facilitating:
    S_layer_D_layer_F_proj = Projection(S_layer_neurons, D_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uef, tau_rec=tau_rec_ef, tau_facil=tau_facil_ef,weight=Exc_Fac_weight*g_S_Layer_D_Layer*r_d*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='esyn')

    # 	- Depressing:
    S_layer_D_layer_D_proj = Projection(S_layer_neurons, D_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Ued, tau_rec=tau_rec_ed, tau_facil=tau_facil_ed,weight=Exc_Dep_weight*g_S_Layer_D_Layer*r_d*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='esyn')

    # 	- Pseudolinear:
    S_layer_D_layer_P_proj = Projection(S_layer_neurons, D_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uep, tau_rec=tau_rec_ep, tau_facil=tau_facil_ep,weight=Exc_Pseudo_weight*g_S_Layer_D_Layer*r_d*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='esyn')
    #	2)  CI -> D layer - 
    # 	- Facilitating:	
    CI_D_layer_F_proj      = Projection(CI_neurons, D_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uif, tau_rec=tau_rec_if, tau_facil=tau_facil_if,weight=Inh_Fac_weight*g_CI_D_Layer*r_d*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='isyn')

    # 	- Depressing:	
    CI_D_layer_D_proj      = Projection(CI_neurons, D_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uid, tau_rec=tau_rec_id, tau_facil=tau_facil_id,weight=Inh_Dep_weight*g_CI_D_Layer*r_d*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='isyn')

    # 	- Pseudolinear:	
    CI_D_layer_P_proj      = Projection(CI_neurons, D_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uip, tau_rec=tau_rec_ip, tau_facil=tau_facil_ip,weight=Inh_Pseudo_weight*g_CI_D_Layer*r_d*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='isyn')

    #	3)  TCR -> D layer - 
    # 	- Depressing:	(Purely Depressing)
    TCR_D_layer_D_proj = Projection(TCR_TC_neurons, D_layer_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Ued, tau_rec=tau_rec_ed, tau_facil=tau_facil_ed,weight=g_TCR_D_Layer*r_d*connectivity_structure, delay=t_d_TC+t_d_syn),
                                                receptor_type='esyn')

    ## CI neurons:
    #	1)  S layer -> CI - 
    # 	- Facilitating:
    S_layer_CI_F_proj = Projection(S_layer_neurons, CI_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uef, tau_rec=tau_rec_ef, tau_facil=tau_facil_ef,weight=Exc_Fac_weight*g_S_Layer_CI*r_ins*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='esyn')

    # 	- Depressing:
    S_layer_CI_D_proj = Projection(S_layer_neurons, CI_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Ued, tau_rec=tau_rec_ed, tau_facil=tau_facil_ed,weight=Exc_Dep_weight*g_S_Layer_CI*r_ins*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='esyn')

    # 	- Pseudolinear:
    S_layer_CI_P_proj = Projection(S_layer_neurons, CI_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uep, tau_rec=tau_rec_ep, tau_facil=tau_facil_ep,weight=Exc_Pseudo_weight*g_S_Layer_CI*r_ins*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='esyn')

    #	2)  M layer -> CI - 
    # 	- Facilitating:
    M_layer_CI_F_proj = Projection(M_layer_neurons, CI_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uef, tau_rec=tau_rec_ef, tau_facil=tau_facil_ef,weight=Exc_Fac_weight*g_M_Layer_CI*r_ins*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='esyn')

    # 	- Depressing:
    M_layer_CI_D_proj = Projection(M_layer_neurons, CI_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Ued, tau_rec=tau_rec_ed, tau_facil=tau_facil_ed,weight=Exc_Dep_weight*g_M_Layer_CI*r_ins*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='esyn')

    # 	- Pseudolinear:
    M_layer_CI_P_proj = Projection(M_layer_neurons, CI_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uep, tau_rec=tau_rec_ep, tau_facil=tau_facil_ep,weight=Exc_Pseudo_weight*g_M_Layer_CI*r_ins*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='esyn')

    #	3)  D layer -> CI - 
    # 	- Facilitating:
    D_layer_CI_F_proj = Projection(D_layer_neurons, CI_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uef, tau_rec=tau_rec_ef, tau_facil=tau_facil_ef,weight=Exc_Fac_weight*g_D_Layer_CI*r_ins*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='esyn')

    # 	- Depressing:
    D_layer_CI_D_proj = Projection(D_layer_neurons, CI_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Ued, tau_rec=tau_rec_ed, tau_facil=tau_facil_ed,weight=Exc_Dep_weight*g_D_Layer_CI*r_ins*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='esyn')

    # 	- Pseudolinear:
    D_layer_CI_P_proj = Projection(D_layer_neurons, CI_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uep, tau_rec=tau_rec_ep, tau_facil=tau_facil_ep,weight=Exc_Pseudo_weight*g_D_Layer_CI*r_ins*connectivity_structure, delay=t_d_l+t_d_syn),
                                                receptor_type='esyn')

    #	4)  TCR -> CI - 
    # 	- Facilitating:
    TCR_CI_F_proj = Projection(TCR_TC_neurons, CI_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uef, tau_rec=tau_rec_ef, tau_facil=tau_facil_ef,weight=Exc_Fac_weight*g_TCR_CI*r_ins*connectivity_structure, delay=t_d_TC+t_d_syn),
                                                receptor_type='esyn')

    # 	- Depressing:
    TCR_CI_D_proj = Projection(TCR_TC_neurons, CI_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Ued, tau_rec=tau_rec_ed, tau_facil=tau_facil_ed,weight=Exc_Dep_weight*g_TCR_CI*r_ins*connectivity_structure, delay=t_d_TC+t_d_syn),
                                                receptor_type='esyn')

    # 	- Pseudolinear:
    TCR_CI_P_proj = Projection(TCR_TC_neurons, CI_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uep, tau_rec=tau_rec_ep, tau_facil=tau_facil_ep,weight=Exc_Pseudo_weight*g_TCR_CI*r_ins*connectivity_structure, delay=t_d_TC+t_d_syn),
                                                receptor_type='esyn')

    ## TRN:
    #	1)  D layer -> TRN - 
    # 	- Facilitating:  (Purely Facilitating)
    D_layer_TRN_F_proj = Projection(D_layer_neurons, TRN_TR_neurons,
                                                #connector=ArrayConnector(connectivity_structure[:,0:TRN_TR_neurons.size]),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uef, tau_rec=tau_rec_ef, tau_facil=tau_facil_ef,weight=g_D_Layer_TRN*r_ret*connectivity_structure[:,0:TRN_TR_neurons.size], delay=t_d_CT+t_d_syn),
                                                receptor_type='esyn')

    #	2)  TCR -> TRN - 
    # 	- Facilitating:
    TCR_TRN_F_proj = Projection(TCR_TC_neurons, TRN_TR_neurons,
                                                #connector=ArrayConnector(connectivity_structure[:,0:TRN_TR_neurons.size]),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uef, tau_rec=tau_rec_ef, tau_facil=tau_facil_ef,weight=Exc_Fac_weight*g_TCR_TRN*r_ret*connectivity_structure[:,0:TRN_TR_neurons.size], delay=t_d_l+t_d_syn),
                                                receptor_type='esyn')	
    # 	- Depressing:
    TCR_TRN_D_proj = Projection(TCR_TC_neurons, TRN_TR_neurons,
                                                #connector=ArrayConnector(connectivity_structure[:,0:TRN_TR_neurons.size]),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Ued, tau_rec=tau_rec_ed, tau_facil=tau_facil_ed,weight=Exc_Dep_weight*g_TCR_TRN*r_ret*connectivity_structure[:,0:TRN_TR_neurons.size], delay=t_d_l+t_d_syn),
                                                receptor_type='esyn')

    # 	- Pseudolinear:
    TCR_TRN_P_proj = Projection(TCR_TC_neurons, TRN_TR_neurons,
                                                #connector=ArrayConnector(connectivity_structure[:,0:TRN_TR_neurons.size]),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uep, tau_rec=tau_rec_ep, tau_facil=tau_facil_ep,weight=Exc_Pseudo_weight*g_TCR_TRN*r_ret*connectivity_structure[:,0:TRN_TR_neurons.size], delay=t_d_l+t_d_syn),
                                                receptor_type='esyn')											

    ## TCR:
    #	1)  D layer -> TCR - 
    # 	- Facilitating:  (Purely Facilitating)
    D_layer_TCR_F_proj = Projection(D_layer_neurons, TCR_TC_neurons,
                                                #connector=ArrayConnector(connectivity_structure),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uef, tau_rec=tau_rec_ef, tau_facil=tau_facil_ef,weight=g_D_Layer_TCR*r_rel*connectivity_structure, delay=t_d_CT+t_d_syn),
                                                receptor_type='esyn')

    #	2)  TRN -> TCR - 
    # 	- Facilitating:
    TRN_TCR_F_proj = Projection(TRN_TR_neurons, TCR_TC_neurons,
                                                #connector=ArrayConnector(connectivity_structure[0:TRN_TR_neurons.size,:]),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uif, tau_rec=tau_rec_if, tau_facil=tau_facil_if,weight=Inh_Fac_weight*g_TRN_TCR*r_rel*connectivity_structure[0:TRN_TR_neurons.size,:], delay=t_d_l+t_d_syn),
                                                receptor_type='isyn')

    # 	- Depressing:
    TRN_TCR_D_proj = Projection(TRN_TR_neurons, TCR_TC_neurons,
                                                #connector=ArrayConnector(connectivity_structure[0:TRN_TR_neurons.size,:]),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uid, tau_rec=tau_rec_id, tau_facil=tau_facil_id,weight=Inh_Dep_weight*g_TRN_TCR*r_rel*connectivity_structure[0:TRN_TR_neurons.size,:], delay=t_d_l+t_d_syn),
                                                receptor_type='isyn')

    # 	- Pseudolinear:
    TRN_TCR_P_proj = Projection(TRN_TR_neurons, TCR_TC_neurons,
                                                #connector=ArrayConnector(connectivity_structure[0:TRN_TR_neurons.size,:]),
                                                connector=AllToAllConnector(allow_self_connections=True),
                                                synapse_type=TsodyksMarkramSynapse(U=Uip, tau_rec=tau_rec_ip, tau_facil=tau_facil_ip,weight=Inh_Pseudo_weight*g_TRN_TCR*r_rel*connectivity_structure[0:TRN_TR_neurons.size,:], delay=t_d_l+t_d_syn),
                                                receptor_type='isyn')
    #end = timer()
    #print('Done with building the network! \nElapsed time:', end-start, 'seconds.')

    end =timer()
    print('Done with building the TCM network! \nElapsed time:', end-start, 'seconds.')
    

    ###--------------------------------------------------- Recording ----------------------------------------------------### 
    # Need to specify before model simulation what we want to record from the model - 
    # Record membrane potentials:
    S_layer_neurons.record(['soma(0.5).v', 'spikes', 'esyn.i'])				# Record the S layer neuron membrane potentials
    M_layer_neurons.record(['soma(0.5).v', 'spikes', 'esyn.i'])				# Record the M layer neuron membrane potentials
    D_layer_neurons.record(['soma(0.5).v', 'spikes', 'esyn.i'])				# Record the D layer neuron membrane potentials
    CI_neurons.record(['soma(0.5).v', 'spikes', 'isyn.i'])					# Record the Cortical Interneuron membrane potentials
    TRN_TR_neurons.record(['soma(0.5).v', 'spikes', 'isyn.i'])				# Record the Thalamic Reticular Nucleus membrane potentials
    TCR_TC_neurons.record(['soma(0.5).v', 'spikes', 'esyn.i'])				# Record the Thalamo-Cortical Relay Neuron membrane potentials



    # Simulate the model for 
    print("\nRunning the isolated thalamo-cortical microcircuit simulation ...")
    start=timer()
    run(simulation_time)
    end = timer()	

    print("\n Simulation Done!\n Total elapsed time:", end-start, "seconds.\n")

    #================= this reset the network time for further simulations, no netwrok structure and recording will be changed =======
    # KO: Not really needed and it fails with newer versions of numpy
    # reset()


    # Save simulation results for postprecessing:
    print("Write the model outputs to .mat files for postprocessing...\n")

    # Write the specified  recorded variables to .mat files (membrane voltages, postsynaptic currents and spike times):
    S_layer_neurons.write_data("Results/S_Layer.mat")
    M_layer_neurons.write_data("Results/M_Layer.mat")
    D_layer_neurons.write_data("Results/D_Layer.mat")
    CI_neurons.write_data("Results/CI_Neurons.mat")
    TCR_TC_neurons.write_data("Results/TCR_Nucleus.mat")
    TRN_TR_neurons.write_data("Results/TR_Nucleus.mat")

    print("Done!")