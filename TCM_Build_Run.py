import datetime
import neuron
from math import pi
from pyNN.neuron import setup, run, end,\
    Population, Projection, AllToAllConnector, TsodyksMarkramSynapse
from pyNN.random import RandomDistribution, NumpyRNG
from Cortical_Basal_Ganglia_Cell_Classes import Izhikevich_Type
import numpy as np
from time import perf_counter
from pathlib import Path
import yaml
import model_values

# Import global variables for generating the streams for the membrane noise
import Global_Variables as GV

# Load ranstream hoc template for generating random streams
h = neuron.h
h.load_file("ranstream.hoc")


def randomise_c_d_values(population):
    neuron_num = population.size

    r = np.random.uniform(0, 1, neuron_num)
    c_value = population.get('c')
    updated_c_values = c_value + 15 * (r ** 2)
    population.set(c=updated_c_values)

    d_value = population.get('d')
    updated_d_values = d_value - 0.6 * (r ** 2)
    population.set(d=updated_d_values)


def randomise_a_b_values(population):
    neuron_num = population.size

    # Add variability to a values
    r = np.random.uniform(0, 1, neuron_num)
    a_value = population.get('a_')
    updated_a_values = a_value + 0.008 * r
    population.set(a_=updated_a_values)

    # Add variability to b values
    b_value = population.get('b')
    updated_b_values = b_value - 0.005 * r
    population.set(b=updated_b_values)


def save_config_to_file(config, filename):
    with open(filename, "w") as f:
        yaml.dump(config, f)


if __name__ == "__main__":

    start_time = perf_counter()
    # initialize simulation
    simulation_timestep = 0.1
    setup(timestep=simulation_timestep, rngseed=3695456)
    simulation_time = 10000.0		# ms
    pop_size = 100
    PD_factor = 800

    print(f"\nSetting simulation time to {simulation_time / 1000} s with time step = {simulation_timestep} ms")

    # Set the global offset variable for the random streams
    GV.random_stream_offset = int(simulation_time * (1.0 / simulation_timestep))

    # Use CVode to calculate i_membrane_ for fast LFP calculation
    cvode = h.CVode()  # cvode is a variable time step methods
    cvode.active(0)

    # Create random distribution for initial cell membrane voltages
    v_init = RandomDistribution('uniform', (-70.0, -60.0))
    parameter_variability = RandomDistribution('uniform', (0, 1))

    # Population Size
    # pop_size = 100
    print("\nNeuronal population size", pop_size)

    # Neuron (and Exponential Synapse) Parameters - Taken from AmirAli's Matlab model: see the supplementary material!
    RS_parameters = model_values.RS_parameters
    IB_parameters = model_values.IB_parameters
    FS_parameters = model_values.FS_parameters
    LTS_parameters = model_values.LTS_parameters
    Rel_TC_parameters = model_values.Rel_TC_parameters
    Ret_parameters = model_values.Ret_parameters

    # ----------------------------------------- Neuron Populations --------------------------------------------------- #
    print("\nAssembling neuronal populations within cortical layers and thalamic nuclei...")

    # Cortical layers and Interneuron Assemblies:
    # --------------------------------------------------- S Layer ---------------------------------------------------- #
    # S layer - Regular Spiking Neurons:
    S_layer_RS_neuron_num = round(0.5 * pop_size)
    S_layer_RS_Population = Population(S_layer_RS_neuron_num, Izhikevich_Type(**RS_parameters), label="S_layer_RS")

    # Set the seed for the random object for parameter variation
    np.random.seed(3695)

    randomise_c_d_values(S_layer_RS_Population)

    # S layer - Intermittent Bursting Neurons
    S_layer_IB_neuron_num = pop_size - S_layer_RS_neuron_num
    S_layer_IB_Population = Population(S_layer_IB_neuron_num, Izhikevich_Type(**IB_parameters), label="S_layer_IB")

    randomise_c_d_values(S_layer_IB_Population)

    # Make assembly for the S layer - Contains RS and IB neurons
    S_layer_neurons = S_layer_RS_Population + S_layer_IB_Population

    # --------------------------------------------------- M Layer ---------------------------------------------------- #
    # M layer - Regular Spiking Neurons:
    M_layer_RS_neuron_num = pop_size
    M_layer_RS_Population = Population(M_layer_RS_neuron_num, Izhikevich_Type(**RS_parameters), label="M_layer_RS")

    randomise_c_d_values(M_layer_RS_Population)

    # Make assembly for the M layer - Contains only RS neurons
    M_layer_neurons = M_layer_RS_Population

    # --------------------------------------------------- D Layer ---------------------------------------------------- #
    # D layer - Regular Spiking Neurons:
    D_layer_RS_neuron_num = round(0.7 * pop_size)
    D_layer_RS_Population = Population(D_layer_RS_neuron_num, Izhikevich_Type(**RS_parameters), label="D_layer_RS")

    randomise_c_d_values(D_layer_RS_Population)

    # D layer - Intermittent Bursting Neurons
    D_layer_IB_neuron_num = pop_size - D_layer_RS_neuron_num
    D_layer_IB_Population = Population(D_layer_IB_neuron_num, Izhikevich_Type(**IB_parameters), label="D_Layer_IB")

    randomise_c_d_values(D_layer_IB_Population)

    # Make assembly for the D layer - Contains RS and IB neurons
    D_layer_neurons = D_layer_RS_Population + D_layer_IB_Population

    # --------------------------------------------------- Cortical Interneurons ---------------------------------------------------- #
    # Cortical Interneurons (CI) - Fast Spiking Interneurons:
    CI_FS_neuron_num = round(0.5 * pop_size)
    CI_FS_Population = Population(CI_FS_neuron_num, Izhikevich_Type(**FS_parameters), label="CI_FS")

    randomise_a_b_values(CI_FS_Population)

    # Cortical Interneurons (CI) - Low-Threshold Spiking Interneurons:
    CI_LTS_neuron_num = pop_size - CI_FS_neuron_num
    CI_LTS_Population = Population(CI_LTS_neuron_num, Izhikevich_Type(**LTS_parameters), label="CI_LTS")

    randomise_a_b_values(CI_LTS_Population)

    # Make assembly for the cortical interneurons - Contains both Fast Spiking and Low-Threshold Spiking Interneurons
    CI_neurons = CI_FS_Population + CI_LTS_Population

    # --------------------------------------------------- Thalamic Reticular Nucleus (TRN) ---------------------------------------------------- #
    TRN_TR_neuron_num = round(0.4 * pop_size)
    TRN_TR_Population = Population(TRN_TR_neuron_num, Izhikevich_Type(**Ret_parameters), label="Th_Reticular")

    randomise_a_b_values(TRN_TR_Population)

    # Make assembly for Thalamic Reticular Nucleus
    TRN_TR_neurons = TRN_TR_Population

    # --------------------------------------------------- Thalamo-Cortical Relay Nucleus (TCR) ---------------------------------------------------- #
    TCR_TC_neuron_num = pop_size
    TCR_TC_Population = Population(TCR_TC_neuron_num, Izhikevich_Type(**Rel_TC_parameters), label="Th_Relay")

    randomise_a_b_values(TCR_TC_Population)

    # Make assembly for Thalamo-Cortical Relay Nucleus
    TCR_TC_neurons = TCR_TC_Population

    # Make assembly of all cells in the thalamo-cortical network
    all_TC_neurons = S_layer_neurons + M_layer_neurons + D_layer_neurons + CI_neurons + TRN_TR_neurons + TCR_TC_neurons

    # ----------------------------------------- Define Connections --------------------------------------------------- #
    # Set up coupling within the network
    print("\nSetting up connections:")
    # Generate random numbers from gaussian distribution for setting up synaptic weights as in Matlab model
    rng = NumpyRNG(seed=8658764)
    weight_uniform_dist = RandomDistribution('uniform', [0, 1], rng=rng)

    r_s = weight_uniform_dist.next(S_layer_neurons.size)
    r_m = weight_uniform_dist.next(M_layer_neurons.size)
    r_d = weight_uniform_dist.next(D_layer_neurons.size)
    r_ins = weight_uniform_dist.next(CI_neurons.size)
    r_ret = weight_uniform_dist.next(TRN_TR_neurons.size)
    r_rel = weight_uniform_dist.next(TCR_TC_neurons.size)

    # All-to-All Connectivity Structure:
    connectivity_structure = np.ones((pop_size, pop_size), np.int_)
    print("Defining coupling distribution; all to all uniformly distributed random coupling.\nSetting up the type of synapses and their distributions within and between each layer and nuclei...")

    # Define the adjacency matrices for each population (each connection is rescaled by max value below in projection definitions)
    r_s = r_s / S_layer_neurons.size  #FK ?
    r_m = r_m / M_layer_neurons.size
    r_d = r_d / D_layer_neurons.size
    r_ins = r_ins / CI_neurons.size
    r_ret = r_ret / TRN_TR_neurons.size
    r_rel = r_rel / TCR_TC_neurons.size

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

    # The synaptic ratios that will be multiplied in the adjacancy matrix calculated based on F synapses:
    Exc_syn_ratio_D = Exc_Dep_weight / Exc_Fac_weight
    Exc_syn_ratio_P = Exc_Pseudo_weight / Exc_Fac_weight
    Inh_syn_ratio_D = Inh_Dep_weight / Inh_Fac_weight
    Inh_syn_ratio_P = Inh_Pseudo_weight / Inh_Fac_weight

    # ---------------------------------------------Synaptic weight values: --------------------------------------------- #
    # PD_factor = 2.5							# Scaling factor for simulating normal conditions
    # PD_factor = 800							# Scaling factor for simulating parkinsonian conditions (Maybe ~8*pop_size)
    print("Factor: ", PD_factor)
    print("Be patient, it takes a while...")
    # Define scaling factor for rescaling original publication synaptic weights for NEURON - more details of current rescaling are described in Izh.mod
    cell_diam = 10.0 / pi
    cell_L = 10.0
    cell_cm = 1.0
    synaptic_rescale_factor = pi * cell_diam * cell_L * cell_cm * 1e-5 / PD_factor

    max_g = model_values.max_g_pd

    # Max Coupling Strengths within each structure
    g_S_Layer_S_Layer = synaptic_rescale_factor * max_g["S"]["S"]
    g_M_Layer_M_Layer = synaptic_rescale_factor * max_g["M"]["M"]
    g_D_Layer_D_Layer = synaptic_rescale_factor * max_g["D"]["D"]
    g_CI_CI = synaptic_rescale_factor * max_g["CI"]["CI"]
    g_TRN_TRN = synaptic_rescale_factor * max_g["TRN"]["TRN"]
    g_TCR_TCR = synaptic_rescale_factor * max_g["TCR"]["TCR"]

    # Max Coupling Strengths between structures
    # Couplings to S Layer
    g_M_Layer_S_Layer = synaptic_rescale_factor * max_g["S"]["M"]
    g_D_Layer_S_Layer = synaptic_rescale_factor * max_g["S"]["D"]
    g_CI_S_Layer = synaptic_rescale_factor * max_g["S"]["CI"]
    g_TRN_S_Layer = synaptic_rescale_factor * max_g["S"]["TRN"]
    g_TCR_S_Layer = synaptic_rescale_factor * max_g["S"]["TCR"]

    # Couplings to M Layer
    g_S_Layer_M_Layer = synaptic_rescale_factor * max_g["M"]["S"]
    g_D_Layer_M_Layer = synaptic_rescale_factor * max_g["M"]["D"]
    g_CI_M_Layer = synaptic_rescale_factor * max_g["M"]["CI"]
    g_TRN_M_Layer = synaptic_rescale_factor * max_g["M"]["TRN"]
    g_TCR_M_Layer = synaptic_rescale_factor * max_g["M"]["TCR"]

    # Couplings to D Layer
    g_S_Layer_D_Layer = synaptic_rescale_factor * max_g["D"]["S"]
    g_M_Layer_D_Layer = synaptic_rescale_factor * max_g["D"]["M"]
    g_CI_D_Layer = synaptic_rescale_factor * max_g["D"]["CI"]
    g_TRN_D_Layer = synaptic_rescale_factor * max_g["D"]["TRN"]
    g_TCR_D_Layer = synaptic_rescale_factor * max_g["D"]["TCR"]

    # Couplings to CI neurons
    g_S_Layer_CI = synaptic_rescale_factor * max_g["CI"]["S"]
    g_M_Layer_CI = synaptic_rescale_factor * max_g["CI"]["M"]
    g_D_Layer_CI = synaptic_rescale_factor * max_g["CI"]["D"]
    g_TRN_CI = synaptic_rescale_factor * max_g["CI"]["TRN"]
    g_TCR_CI = synaptic_rescale_factor * max_g["CI"]["TCR"]

    # Couplings to TRN neurons
    g_S_Layer_TRN = synaptic_rescale_factor * max_g["TRN"]["S"]
    g_M_Layer_TRN = synaptic_rescale_factor * max_g["TRN"]["M"]
    g_D_Layer_TRN = synaptic_rescale_factor * max_g["TRN"]["D"]
    g_CI_TRN = synaptic_rescale_factor * max_g["TRN"]["CI"]
    g_TCR_TRN = synaptic_rescale_factor * max_g["TRN"]["TCR"]

    # Couplings to TCR neurons
    g_S_Layer_TCR = synaptic_rescale_factor * max_g["TCR"]["S"]
    g_M_Layer_TCR = synaptic_rescale_factor * max_g["TCR"]["M"]
    g_D_Layer_TCR = synaptic_rescale_factor * max_g["TCR"]["D"]
    g_CI_TCR = synaptic_rescale_factor * max_g["TCR"]["CI"]
    g_TRN_TCR = synaptic_rescale_factor * max_g["TCR"]["TRN"]

    # Synaptic Delays
    t_d_l = 8		# Time delay between the layers in cortex and nuclei in thalamus (ms)
    t_d_wl = 1		# Time delay within a structure (ms)
    t_d_TC = 15		# Time delay between the thalamus and cortex (ms)
    t_d_CT = 20		# Time delay between the cortex and thalamus (ms)
    t_d_syn = 1		# Synaptic transmission delay (ms - fixed for all synapses in the TCM)

    # -------------------------------------------- Coupling within populations -------------------------------------------- #
    # S Layer:
    # 	- Facilitating:
    S_layer_S_layer_F_proj = Projection(S_layer_neurons, S_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Uif, tau_rec=tau_rec_if, tau_facil=tau_facil_if,weight=Inh_Fac_weight*g_S_Layer_S_Layer*r_s*connectivity_structure, delay=t_d_wl+t_d_syn),
                                        receptor_type='isyn')		
    # 	- Depressing:
    S_layer_S_layer_D_proj = Projection(S_layer_neurons, S_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Uid, tau_rec=tau_rec_id, tau_facil=tau_facil_id,weight=Inh_Dep_weight*g_S_Layer_S_Layer*r_s*connectivity_structure, delay=t_d_wl+t_d_syn),
                                        receptor_type='isyn')	
    # 	- Pseudolinear:
    S_layer_S_layer_P_proj = Projection(S_layer_neurons, S_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Uip, tau_rec=tau_rec_ip, tau_facil=tau_facil_ip,weight=Inh_Pseudo_weight*g_S_Layer_S_Layer*r_s*connectivity_structure, delay=t_d_wl+t_d_syn),
                                        receptor_type='isyn')			

    # M Layer:
    # 	- Facilitating:
    M_layer_M_layer_F_proj = Projection(M_layer_neurons, M_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Uif, tau_rec=tau_rec_if, tau_facil=tau_facil_if,weight=Inh_Fac_weight*g_M_Layer_M_Layer*r_m*connectivity_structure, delay=t_d_wl+t_d_syn),
                                        receptor_type='isyn')		
    # 	- Depressing:
    M_layer_M_layer_D_proj = Projection(M_layer_neurons, M_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Uid, tau_rec=tau_rec_id, tau_facil=tau_facil_id,weight=Inh_Dep_weight*g_M_Layer_M_Layer*r_m*connectivity_structure, delay=t_d_wl+t_d_syn),
                                        receptor_type='isyn')	
    # 	- Pseudolinear:
    M_layer_M_layer_P_proj = Projection(M_layer_neurons, M_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Uip, tau_rec=tau_rec_ip, tau_facil=tau_facil_ip,weight=Inh_Pseudo_weight*g_M_Layer_M_Layer*r_m*connectivity_structure, delay=t_d_wl+t_d_syn),
                                        receptor_type='isyn')	

    # D Layer:
    # 	- Facilitating:
    D_layer_D_layer_F_proj = Projection(D_layer_neurons, D_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Uif, tau_rec=tau_rec_if, tau_facil=tau_facil_if,weight=Inh_Fac_weight*g_D_Layer_D_Layer*r_d*connectivity_structure, delay=t_d_wl+t_d_syn),
                                        receptor_type='isyn')		
    # 	- Depressing:
    D_layer_D_layer_D_proj = Projection(D_layer_neurons, D_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Uid, tau_rec=tau_rec_id, tau_facil=tau_facil_id,weight=Inh_Dep_weight*g_D_Layer_D_Layer*r_d*connectivity_structure, delay=t_d_wl+t_d_syn),
                                        receptor_type='isyn')	
    # 	- Pseudolinear:
    D_layer_D_layer_P_proj = Projection(D_layer_neurons, D_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Uip, tau_rec=tau_rec_ip, tau_facil=tau_facil_ip,weight=Inh_Pseudo_weight*g_D_Layer_D_Layer*r_d*connectivity_structure, delay=t_d_wl+t_d_syn),
                                        receptor_type='isyn')

    # CI Neurons:
    # 	- Facilitating:
    CI_CI_F_proj = Projection(CI_neurons, CI_neurons,
                              connector=AllToAllConnector(allow_self_connections=True),
                              synapse_type=TsodyksMarkramSynapse(U=Uif, tau_rec=tau_rec_if, tau_facil=tau_facil_if,weight=Inh_Fac_weight*g_CI_CI*r_ins*connectivity_structure, delay=t_d_wl+t_d_syn),
                              receptor_type='isyn')

    # 	- Depressing:
    CI_CI_D_proj = Projection(CI_neurons, CI_neurons,
                              connector=AllToAllConnector(allow_self_connections=True),
                              synapse_type=TsodyksMarkramSynapse(U=Uid, tau_rec=tau_rec_id, tau_facil=tau_facil_id,weight=Inh_Dep_weight*g_CI_CI*r_ins*connectivity_structure, delay=t_d_wl+t_d_syn),
                              receptor_type='isyn')

    # 	- Pseudolinear:
    CI_CI_P_proj = Projection(CI_neurons, CI_neurons,
                              connector=AllToAllConnector(allow_self_connections=True),
                              synapse_type=TsodyksMarkramSynapse(U=Uip, tau_rec=tau_rec_ip, tau_facil=tau_facil_ip,weight=Inh_Pseudo_weight*g_CI_CI*r_ins*connectivity_structure, delay=t_d_wl+t_d_syn),
                              receptor_type='isyn')

    # TRN Neurons:
    # 	- Facilitating:
    TRN_TRN_F_proj = Projection(TRN_TR_neurons, TRN_TR_neurons,
                                connector=AllToAllConnector(allow_self_connections=True),
                                synapse_type=TsodyksMarkramSynapse(U=Uif, tau_rec=tau_rec_if, tau_facil=tau_facil_if,weight=Inh_Fac_weight*g_TRN_TRN*r_ret*connectivity_structure[0:TRN_TR_neurons.size,0:TRN_TR_neurons.size], delay=t_d_wl+t_d_syn),
                                receptor_type='isyn')

    # 	- Depressing:
    TRN_TRN_D_proj = Projection(TRN_TR_neurons, TRN_TR_neurons,
                                connector=AllToAllConnector(allow_self_connections=True),
                                synapse_type=TsodyksMarkramSynapse(U=Uid, tau_rec=tau_rec_id, tau_facil=tau_facil_id,weight=Inh_Dep_weight*g_TRN_TRN*r_ret*connectivity_structure[0:TRN_TR_neurons.size,0:TRN_TR_neurons.size], delay=t_d_wl+t_d_syn),
                                receptor_type='isyn')

    # 	- Pseudolinear:
    TRN_TRN_P_proj = Projection(TRN_TR_neurons, TRN_TR_neurons,
                                connector=AllToAllConnector(allow_self_connections=True),
                                synapse_type=TsodyksMarkramSynapse(U=Uip, tau_rec=tau_rec_ip, tau_facil=tau_facil_ip,weight=Inh_Pseudo_weight*g_TRN_TRN*r_ret*connectivity_structure[0:TRN_TR_neurons.size,0:TRN_TR_neurons.size], delay=t_d_wl+t_d_syn),
                                receptor_type='isyn')

    # -------------------------------------------- Coupling between populations - projections -------------------------------------------- #
    # S layer:
    #   1)  M layer -> S layer
    # 	- Facilitating:
    M_layer_S_layer_F_proj = Projection(M_layer_neurons, S_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Uef, tau_rec=tau_rec_ef, tau_facil=tau_facil_ef,weight=Exc_Fac_weight*g_M_Layer_S_Layer*r_s*connectivity_structure, delay=t_d_l+t_d_syn),
                                        receptor_type='esyn')

    # 	- Depressing:
    M_layer_S_layer_D_proj = Projection(M_layer_neurons, S_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Ued, tau_rec=tau_rec_ed, tau_facil=tau_facil_ed,weight=Exc_Dep_weight*g_M_Layer_S_Layer*r_s*connectivity_structure, delay=t_d_l+t_d_syn),
                                        receptor_type='esyn')

    # 	- Pseudolinear:
    M_layer_S_layer_P_proj = Projection(M_layer_neurons, S_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Uep, tau_rec=tau_rec_ep, tau_facil=tau_facil_ep,weight=Exc_Pseudo_weight*g_M_Layer_S_Layer*r_s*connectivity_structure, delay=t_d_l+t_d_syn),
                                        receptor_type='esyn')

    #   2)  D layer -> S layer
    # 	- Facilitating:
    D_layer_S_layer_F_proj = Projection(D_layer_neurons, S_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Uef, tau_rec=tau_rec_ef, tau_facil=tau_facil_ef,weight=Exc_Fac_weight*g_D_Layer_S_Layer*r_s*connectivity_structure, delay=t_d_l+t_d_syn),
                                        receptor_type='esyn')

    # 	- Depressing:
    D_layer_S_layer_D_proj = Projection(D_layer_neurons, S_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Ued, tau_rec=tau_rec_ed, tau_facil=tau_facil_ed,weight=Exc_Dep_weight*g_D_Layer_S_Layer*r_s*connectivity_structure, delay=t_d_l+t_d_syn),
                                        receptor_type='esyn')

    # 	- Pseudolinear:
    D_layer_S_layer_P_proj = Projection(D_layer_neurons, S_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Uep, tau_rec=tau_rec_ep, tau_facil=tau_facil_ep,weight=Exc_Pseudo_weight*g_D_Layer_S_Layer*r_s*connectivity_structure, delay=t_d_l+t_d_syn),
                                        receptor_type='esyn')

    #   3)  CI neurons -> S layer
    # 	- Facilitating:
    CI_S_layer_F_proj 	   = Projection(CI_neurons, S_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Uif, tau_rec=tau_rec_if, tau_facil=tau_facil_if,weight=Inh_Fac_weight*g_CI_S_Layer*r_s*connectivity_structure, delay=t_d_l+t_d_syn),
                                        receptor_type='isyn')

    # 	- Depressing:
    CI_S_layer_D_proj 	   = Projection(CI_neurons, S_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Uid, tau_rec=tau_rec_id, tau_facil=tau_facil_id,weight=Inh_Dep_weight*g_CI_S_Layer*r_s*connectivity_structure, delay=t_d_l+t_d_syn),
                                        receptor_type='isyn')

    # 	- Pseudolinear:
    CI_S_layer_P_proj 	   = Projection(CI_neurons, S_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Uip, tau_rec=tau_rec_ip, tau_facil=tau_facil_ip,weight=Inh_Pseudo_weight*g_CI_S_Layer*r_s*connectivity_structure, delay=t_d_l+t_d_syn),
                                        receptor_type='isyn')

    # M layer:
    #   1)  S layer -> M layer
    # 	- Facilitating:
    S_layer_M_layer_F_proj = Projection(S_layer_neurons, M_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Uef, tau_rec=tau_rec_ef, tau_facil=tau_facil_ef,weight=Exc_Fac_weight*g_S_Layer_M_Layer*r_m*connectivity_structure, delay=t_d_l+t_d_syn),
                                        receptor_type='esyn')

    # 	- Depressing:
    S_layer_M_layer_D_proj = Projection(S_layer_neurons, M_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Ued, tau_rec=tau_rec_ed, tau_facil=tau_facil_ed,weight=Exc_Dep_weight*g_S_Layer_M_Layer*r_m*connectivity_structure, delay=t_d_l+t_d_syn),
                                        receptor_type='esyn')

    # 	- Pseudolinear:
    S_layer_M_layer_P_proj = Projection(S_layer_neurons, M_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Uep, tau_rec=tau_rec_ep, tau_facil=tau_facil_ep,weight=Exc_Pseudo_weight*g_S_Layer_M_Layer*r_m*connectivity_structure, delay=t_d_l+t_d_syn),
                                        receptor_type='esyn')

    #   2)  CI Neurons -> M layer
    # 	- Facilitating:
    CI_M_layer_F_proj = Projection(CI_neurons, M_layer_neurons,
                                   connector=AllToAllConnector(allow_self_connections=True),
                                   synapse_type=TsodyksMarkramSynapse(U=Uif, tau_rec=tau_rec_if, tau_facil=tau_facil_if,weight=Inh_Fac_weight*g_CI_M_Layer*r_m*connectivity_structure, delay=t_d_l+t_d_syn),
                                   receptor_type='isyn')

    # 	- Depressing:
    CI_M_layer_D_proj = Projection(CI_neurons, M_layer_neurons,
                                   connector=AllToAllConnector(allow_self_connections=True),
                                   synapse_type=TsodyksMarkramSynapse(U=Uid, tau_rec=tau_rec_id, tau_facil=tau_facil_id,weight=Inh_Dep_weight*g_CI_M_Layer*r_m*connectivity_structure, delay=t_d_l+t_d_syn),
                                   receptor_type='isyn')

    # 	- Pseudolinear:
    CI_M_layer_P_proj = Projection(CI_neurons, M_layer_neurons,
                                   connector=AllToAllConnector(allow_self_connections=True),
                                   synapse_type=TsodyksMarkramSynapse(U=Uip, tau_rec=tau_rec_ip, tau_facil=tau_facil_ip,weight=Inh_Pseudo_weight*g_CI_M_Layer*r_m*connectivity_structure, delay=t_d_l+t_d_syn),
                                   receptor_type='isyn')

    ## D layer:
    #	1)  S layer -> D layer
    # 	- Facilitating:
    S_layer_D_layer_F_proj = Projection(S_layer_neurons, D_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Uef, tau_rec=tau_rec_ef, tau_facil=tau_facil_ef,weight=Exc_Fac_weight*g_S_Layer_D_Layer*r_d*connectivity_structure, delay=t_d_l+t_d_syn),
                                        receptor_type='esyn')

    # 	- Depressing:
    S_layer_D_layer_D_proj = Projection(S_layer_neurons, D_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Ued, tau_rec=tau_rec_ed, tau_facil=tau_facil_ed,weight=Exc_Dep_weight*g_S_Layer_D_Layer*r_d*connectivity_structure, delay=t_d_l+t_d_syn),
                                        receptor_type='esyn')

    # 	- Pseudolinear:
    S_layer_D_layer_P_proj = Projection(S_layer_neurons, D_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Uep, tau_rec=tau_rec_ep, tau_facil=tau_facil_ep,weight=Exc_Pseudo_weight*g_S_Layer_D_Layer*r_d*connectivity_structure, delay=t_d_l+t_d_syn),
                                        receptor_type='esyn')
    #   2)  CI -> D layer
    # 	- Facilitating:
    CI_D_layer_F_proj      = Projection(CI_neurons, D_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Uif, tau_rec=tau_rec_if, tau_facil=tau_facil_if,weight=Inh_Fac_weight*g_CI_D_Layer*r_d*connectivity_structure, delay=t_d_l+t_d_syn),
                                        receptor_type='isyn')

    # 	- Depressing:
    CI_D_layer_D_proj      = Projection(CI_neurons, D_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Uid, tau_rec=tau_rec_id, tau_facil=tau_facil_id,weight=Inh_Dep_weight*g_CI_D_Layer*r_d*connectivity_structure, delay=t_d_l+t_d_syn),
                                        receptor_type='isyn')

    # 	- Pseudolinear:
    CI_D_layer_P_proj      = Projection(CI_neurons, D_layer_neurons,
                                        connector=AllToAllConnector(allow_self_connections=True),
                                        synapse_type=TsodyksMarkramSynapse(U=Uip, tau_rec=tau_rec_ip, tau_facil=tau_facil_ip,weight=Inh_Pseudo_weight*g_CI_D_Layer*r_d*connectivity_structure, delay=t_d_l+t_d_syn),
                                        receptor_type='isyn')

    #   3)  TCR -> D layer
    # 	- Depressing:	(Purely Depressing)
    TCR_D_layer_D_proj = Projection(TCR_TC_neurons, D_layer_neurons,
                                    connector=AllToAllConnector(allow_self_connections=True),
                                    synapse_type=TsodyksMarkramSynapse(U=Ued, tau_rec=tau_rec_ed, tau_facil=tau_facil_ed,weight=g_TCR_D_Layer*r_d*connectivity_structure, delay=t_d_TC+t_d_syn),
                                    receptor_type='esyn')

    # CI neurons:
    #   1)  S layer -> CI
    # 	- Facilitating:
    S_layer_CI_F_proj = Projection(S_layer_neurons, CI_neurons,
                                   connector=AllToAllConnector(allow_self_connections=True),
                                   synapse_type=TsodyksMarkramSynapse(U=Uef, tau_rec=tau_rec_ef, tau_facil=tau_facil_ef,weight=Exc_Fac_weight*g_S_Layer_CI*r_ins*connectivity_structure, delay=t_d_l+t_d_syn),
                                   receptor_type='esyn')

    # 	- Depressing:
    S_layer_CI_D_proj = Projection(S_layer_neurons, CI_neurons,
                                   connector=AllToAllConnector(allow_self_connections=True),
                                   synapse_type=TsodyksMarkramSynapse(U=Ued, tau_rec=tau_rec_ed, tau_facil=tau_facil_ed,weight=Exc_Dep_weight*g_S_Layer_CI*r_ins*connectivity_structure, delay=t_d_l+t_d_syn),
                                   receptor_type='esyn')

    # 	- Pseudolinear:
    S_layer_CI_P_proj = Projection(S_layer_neurons, CI_neurons,
                                   connector=AllToAllConnector(allow_self_connections=True),
                                   synapse_type=TsodyksMarkramSynapse(U=Uep, tau_rec=tau_rec_ep, tau_facil=tau_facil_ep,weight=Exc_Pseudo_weight*g_S_Layer_CI*r_ins*connectivity_structure, delay=t_d_l+t_d_syn),
                                   receptor_type='esyn')

    #   2)  M layer -> CI
    # 	- Facilitating:
    M_layer_CI_F_proj = Projection(M_layer_neurons, CI_neurons,
                                   connector=AllToAllConnector(allow_self_connections=True),
                                   synapse_type=TsodyksMarkramSynapse(U=Uef, tau_rec=tau_rec_ef, tau_facil=tau_facil_ef,weight=Exc_Fac_weight*g_M_Layer_CI*r_ins*connectivity_structure, delay=t_d_l+t_d_syn),
                                   receptor_type='esyn')

    # 	- Depressing:
    M_layer_CI_D_proj = Projection(M_layer_neurons, CI_neurons,
                                   connector=AllToAllConnector(allow_self_connections=True),
                                   synapse_type=TsodyksMarkramSynapse(U=Ued, tau_rec=tau_rec_ed, tau_facil=tau_facil_ed,weight=Exc_Dep_weight*g_M_Layer_CI*r_ins*connectivity_structure, delay=t_d_l+t_d_syn),
                                   receptor_type='esyn')

    # 	- Pseudolinear:
    M_layer_CI_P_proj = Projection(M_layer_neurons, CI_neurons,
                                   connector=AllToAllConnector(allow_self_connections=True),
                                   synapse_type=TsodyksMarkramSynapse(U=Uep, tau_rec=tau_rec_ep, tau_facil=tau_facil_ep,weight=Exc_Pseudo_weight*g_M_Layer_CI*r_ins*connectivity_structure, delay=t_d_l+t_d_syn),
                                   receptor_type='esyn')

    #   3)  D layer -> CI
    # 	- Facilitating:
    D_layer_CI_F_proj = Projection(D_layer_neurons, CI_neurons,
                                   connector=AllToAllConnector(allow_self_connections=True),
                                   synapse_type=TsodyksMarkramSynapse(U=Uef, tau_rec=tau_rec_ef, tau_facil=tau_facil_ef,weight=Exc_Fac_weight*g_D_Layer_CI*r_ins*connectivity_structure, delay=t_d_l+t_d_syn),
                                   receptor_type='esyn')

    # 	- Depressing:
    D_layer_CI_D_proj = Projection(D_layer_neurons, CI_neurons,
                                   connector=AllToAllConnector(allow_self_connections=True),
                                   synapse_type=TsodyksMarkramSynapse(U=Ued, tau_rec=tau_rec_ed, tau_facil=tau_facil_ed,weight=Exc_Dep_weight*g_D_Layer_CI*r_ins*connectivity_structure, delay=t_d_l+t_d_syn),
                                   receptor_type='esyn')

    # 	- Pseudolinear:
    D_layer_CI_P_proj = Projection(D_layer_neurons, CI_neurons,
                                   connector=AllToAllConnector(allow_self_connections=True),
                                   synapse_type=TsodyksMarkramSynapse(U=Uep, tau_rec=tau_rec_ep, tau_facil=tau_facil_ep,weight=Exc_Pseudo_weight*g_D_Layer_CI*r_ins*connectivity_structure, delay=t_d_l+t_d_syn),
                                   receptor_type='esyn')

    #   4)  TCR -> CI
    # 	- Facilitating:
    TCR_CI_F_proj = Projection(TCR_TC_neurons, CI_neurons,
                               connector=AllToAllConnector(allow_self_connections=True),
                               synapse_type=TsodyksMarkramSynapse(U=Uef, tau_rec=tau_rec_ef, tau_facil=tau_facil_ef,weight=Exc_Fac_weight*g_TCR_CI*r_ins*connectivity_structure, delay=t_d_TC+t_d_syn),
                               receptor_type='esyn')

    # 	- Depressing:
    TCR_CI_D_proj = Projection(TCR_TC_neurons, CI_neurons,
                               connector=AllToAllConnector(allow_self_connections=True),
                               synapse_type=TsodyksMarkramSynapse(U=Ued, tau_rec=tau_rec_ed, tau_facil=tau_facil_ed,weight=Exc_Dep_weight*g_TCR_CI*r_ins*connectivity_structure, delay=t_d_TC+t_d_syn),
                               receptor_type='esyn')

    # 	- Pseudolinear:
    TCR_CI_P_proj = Projection(TCR_TC_neurons, CI_neurons,
                               connector=AllToAllConnector(allow_self_connections=True),
                               synapse_type=TsodyksMarkramSynapse(U=Uep, tau_rec=tau_rec_ep, tau_facil=tau_facil_ep,weight=Exc_Pseudo_weight*g_TCR_CI*r_ins*connectivity_structure, delay=t_d_TC+t_d_syn),
                               receptor_type='esyn')

    # TRN:
    #   1)  D layer -> TRN
    # 	- Facilitating:  (Purely Facilitating)
    D_layer_TRN_F_proj = Projection(D_layer_neurons, TRN_TR_neurons,
                                    connector=AllToAllConnector(allow_self_connections=True),
                                    synapse_type=TsodyksMarkramSynapse(U=Uef, tau_rec=tau_rec_ef, tau_facil=tau_facil_ef,weight=g_D_Layer_TRN*r_ret*connectivity_structure[:,0:TRN_TR_neurons.size], delay=t_d_CT+t_d_syn),
                                    receptor_type='esyn')

    #   2)  TCR -> TRN
    # 	- Facilitating:
    TCR_TRN_F_proj = Projection(TCR_TC_neurons, TRN_TR_neurons,
                                connector=AllToAllConnector(allow_self_connections=True),
                                synapse_type=TsodyksMarkramSynapse(U=Uef, tau_rec=tau_rec_ef, tau_facil=tau_facil_ef,weight=Exc_Fac_weight*g_TCR_TRN*r_ret*connectivity_structure[:,0:TRN_TR_neurons.size], delay=t_d_l+t_d_syn),
                                receptor_type='esyn')
    # 	- Depressing:
    TCR_TRN_D_proj = Projection(TCR_TC_neurons, TRN_TR_neurons,
                                connector=AllToAllConnector(allow_self_connections=True),
                                synapse_type=TsodyksMarkramSynapse(U=Ued, tau_rec=tau_rec_ed, tau_facil=tau_facil_ed,weight=Exc_Dep_weight*g_TCR_TRN*r_ret*connectivity_structure[:,0:TRN_TR_neurons.size], delay=t_d_l+t_d_syn),
                                receptor_type='esyn')

    # 	- Pseudolinear:
    TCR_TRN_P_proj = Projection(TCR_TC_neurons, TRN_TR_neurons,
                                connector=AllToAllConnector(allow_self_connections=True),
                                synapse_type=TsodyksMarkramSynapse(U=Uep, tau_rec=tau_rec_ep, tau_facil=tau_facil_ep,weight=Exc_Pseudo_weight*g_TCR_TRN*r_ret*connectivity_structure[:,0:TRN_TR_neurons.size], delay=t_d_l+t_d_syn),
                                receptor_type='esyn')											

    # TCR:
    #   1)  D layer -> TCR
    # 	- Facilitating:  (Purely Facilitating)
    D_layer_TCR_F_proj = Projection(D_layer_neurons, TCR_TC_neurons,
                                    connector=AllToAllConnector(allow_self_connections=True),
                                    synapse_type=TsodyksMarkramSynapse(U=Uef, tau_rec=tau_rec_ef, tau_facil=tau_facil_ef,weight=g_D_Layer_TCR*r_rel*connectivity_structure, delay=t_d_CT+t_d_syn),
                                    receptor_type='esyn')

    #   2)  TRN -> TCR
    # 	- Facilitating:
    TRN_TCR_F_proj = Projection(TRN_TR_neurons, TCR_TC_neurons,
                                connector=AllToAllConnector(allow_self_connections=True),
                                synapse_type=TsodyksMarkramSynapse(U=Uif, tau_rec=tau_rec_if, tau_facil=tau_facil_if,weight=Inh_Fac_weight*g_TRN_TCR*r_rel*connectivity_structure[0:TRN_TR_neurons.size,:], delay=t_d_l+t_d_syn),
                                receptor_type='isyn')

    # 	- Depressing:
    TRN_TCR_D_proj = Projection(TRN_TR_neurons, TCR_TC_neurons,
                                connector=AllToAllConnector(allow_self_connections=True),
                                synapse_type=TsodyksMarkramSynapse(U=Uid, tau_rec=tau_rec_id, tau_facil=tau_facil_id,weight=Inh_Dep_weight*g_TRN_TCR*r_rel*connectivity_structure[0:TRN_TR_neurons.size,:], delay=t_d_l+t_d_syn),
                                receptor_type='isyn')

    # 	- Pseudolinear:
    TRN_TCR_P_proj = Projection(TRN_TR_neurons, TCR_TC_neurons,
                                connector=AllToAllConnector(allow_self_connections=True),
                                synapse_type=TsodyksMarkramSynapse(U=Uip, tau_rec=tau_rec_ip, tau_facil=tau_facil_ip,weight=Inh_Pseudo_weight*g_TRN_TCR*r_rel*connectivity_structure[0:TRN_TR_neurons.size,:], delay=t_d_l+t_d_syn),
                                receptor_type='isyn')

    end_time = perf_counter()
    print(f"Done with building the TCM network! \nElapsed time: {end_time - start_time} seconds.")

    # --------------------------------------------------- Recording ---------------------------------------------------- #
    # Record membrane potentials, spikes and synaptic currents:
    S_layer_neurons.record(['soma(0.5).v', 'spikes', 'esyn.i'])
    M_layer_neurons.record(['soma(0.5).v', 'spikes', 'esyn.i'])
    D_layer_neurons.record(['soma(0.5).v', 'spikes', 'esyn.i'])
    CI_neurons.record(['soma(0.5).v', 'spikes', 'isyn.i'])
    TRN_TR_neurons.record(['soma(0.5).v', 'spikes', 'isyn.i'])
    TCR_TC_neurons.record(['soma(0.5).v', 'spikes', 'esyn.i'])

    population_list = [
        S_layer_RS_Population,
        S_layer_IB_Population,
        M_layer_RS_Population,
        D_layer_RS_Population,
        D_layer_IB_Population,
        CI_FS_Population,
        CI_LTS_Population,
        TRN_TR_Population,
        TCR_TC_Population
    ]

    config_dict = {
        "RS_parameters": RS_parameters,
        "IB_parameters": IB_parameters,
        "FS_parameters": FS_parameters,
        "LTS_parameters": LTS_parameters,
        "Relay_TC_parameters": Rel_TC_parameters,
        "Reticular_parameters": Ret_parameters,
        "Population_size": {pop.label: pop.size for pop in population_list},
        "Max_g": max_g,
    }

    # Simulate the model
    print("\nRunning the isolated thalamo-cortical microcircuit simulation ...")
    start_time = perf_counter()
    run(simulation_time)
    end_time = perf_counter()

    print("\n Simulation Done!")
    print(f"Total elapsed time: {end_time - start_time} seconds.\n")

    # Save simulation results for postprecessing:
    print("Write the model outputs to .mat files for postprocessing...\n")
    current_datetime = datetime.datetime.now()
    id_number = 0
    protect_overwrite = True

    while protect_overwrite:
        datetime_string = f"{current_datetime.year:04}{current_datetime.month:02}{current_datetime.day:02}_{current_datetime.hour:02}{current_datetime.minute:02}{current_datetime.second:02}_{id_number:02}"
        output_dir = Path("Results") / datetime_string
        if output_dir.exists():
            id_number += 1
        else:
            protect_overwrite = False
    print(f"Output directory: {output_dir.name}")

    # Write the specified recorded variables to .mat files
    S_layer_neurons.write_data(str(output_dir / "S_Layer.mat"))
    M_layer_neurons.write_data(str(output_dir / "M_Layer.mat"))
    D_layer_neurons.write_data(str(output_dir / "D_Layer.mat"))
    CI_neurons.write_data(str(output_dir / "CI_Neurons.mat"))
    TCR_TC_neurons.write_data(str(output_dir / "TCR_Nucleus.mat"))
    TRN_TR_neurons.write_data(str(output_dir / "TR_Nucleus.mat"))

    save_config_to_file(config_dict, output_dir / "config.yml")

    print("Done!")
    end()
