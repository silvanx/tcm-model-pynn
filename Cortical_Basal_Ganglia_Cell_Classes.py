# -*- coding: utf-8 -*-
""" ------------------------------------------------------------------------------------
	Cortical Basal Ganglia Neurons: file containing classes for defining network neurons
	------------------------------------------------------------------------------------
	
								Model References 
	------------------------------------------------------------------------------------
	Cortical Pyramical Cell Soma:
	Pospischil, M., Toledo-Rodriguez, M., Monier, C., Piwkowska, Z., 
	Bal, T., Frégnac, Y., Markram, H. and Destexhe, A., 2008. 
	"Minimal Hodgkin–Huxley type models for different classes of 
	cortical and thalamic neurons." 
	Biological cybernetics, 99(4-5), pp.427-441.
	
	Cortical Pyramidal Cell Axon: 
	Foust, A.J., Yu, Y., Popovic, M., Zecevic, D. and McCormick, D.A., 
	2011. "Somatic membrane potential and Kv1 channels control spike 
	repolarization in cortical axon collaterals and presynaptic boutons." 
	Journal of Neuroscience, 31(43), pp.15490-15498.
	
	Cortical Interneurons:
	Pospischil, M., Toledo-Rodriguez, M., Monier, C., Piwkowska, Z., 
	Bal, T., Frégnac, Y., Markram, H. and Destexhe, A., 2008. 
	"Minimal Hodgkin–Huxley type models for different classes of 
	cortical and thalamic neurons." 
	Biological cybernetics, 99(4-5), pp.427-441.
  
	STN Neurons:
	Otsuka, T., Abe, T., Tsukagawa, T. and Song, W.J., 2004. 
	"Conductance-based model of the voltage-dependent generation 
	of a plateau potential in subthalamic neurons."
	Journal of neurophysiology, 92(1), pp.255-264.
  
	GP Neurons:
	Terman, D., Rubin, J.E., Yew, A.C. and Wilson, C.J., 2002. 
	"Activity patterns in a model for the subthalamopallidal 
	network of the basal ganglia." 
	Journal of Neuroscience, 22(7), pp.2963-2976.
	
	Thalamic Neurons:
	Rubin, J.E. and Terman, D., 2004. High frequency stimulation of 
	the subthalamic nucleus eliminates pathological thalamic rhythmicity 
	in a computational model. Journal of computational neuroscience, 
	16(3), pp.211-235.
	
	Implemented by John Fleming - john.fleming@ucdconnect.ie - 06/12/18
	
	Edits: 16-01-19: Created classes for cell models so they can be 
					 utilized in PyNN.

	Created on Tues Jan 15 12:51:26 2019

"""

from math import pi
from neuron import h
from nrnutils import Mechanism, Section
from pyNN.neuron import NativeCellType
from pyNN.parameters import Sequence
import numpy as np
from scipy import signal

# Import global variables for generating the streams for the membrane noise
import Global_Variables as GV

try:
    reduce
except NameError:
    from functools import reduce

def _new_property(obj_hierarchy, attr_name):
    """
    Returns a new property, mapping attr_name to obj_hierarchy.attr_name.

    For example, suppose that an object of class A has an attribute b which
    itself has an attribute c which itself has an attribute d. Then placing
        e = _new_property('b.c', 'd')
    in the class definition of A makes A.e an alias for A.b.c.d
    """

    def set(self, value):
        obj = reduce(getattr, [self] + obj_hierarchy.split('.'))
        setattr(obj, attr_name, value)

    def get(self):
        obj = reduce(getattr, [self] + obj_hierarchy.split('.'))
        return getattr(obj, attr_name)
    return property(fset=set, fget=get)

class Izhikevich(object):
    
    def __init__(self, **parameters):
        
        # Create single compartment cell section, i.e. soma section
        self.soma = Section(
            L=parameters['L'],
            diam=parameters['diam'],
            nseg=parameters['nseg'],
            Ra=parameters['Ra'],
            cm=parameters['cm'],
            mechanisms=[]
            )
        
        # Insert the Izhikevich mechanism
        self.izh = h.Izh(0.5, sec=self.soma)
        self.izh.a = parameters['a_']
        self.izh.b = parameters['b']
        self.izh.c = parameters['c']
        self.izh.d = parameters['d']
        self.izh.thresh = parameters['thresh']
        self.izh.dur = 1e9
        self.izh.amp = parameters['bias_current_amp']
        
        
        # Add threshold noise to the neuron
        # r = h.Random()
        # r.Random123(GV.stream_index * GV.random_stream_offset + 1, 1, 0)
        # GV.thresh_rslist.append(r)
        # GV.thresh_rslist[-1].normal(parameters['thresh'], parameters['thresh_noise']) 	# mean 30.0, variance 0.5
        # GV.thresh_rslist[-1].play(self.izh._ref_thresh)
        
        
        # Generate Membrane noise (and the required independent random stream)
        # Append random stream for gaussian membrane noise mechanism to global list
        GV.rslist.append(h.RandomStream(GV.stream_index, 0, 0, GV.random_stream_offset))
        GV.stream_index = GV.stream_index + 1				# Need to increment the global stream index variable
        GV.rslist[-1].r.normal(0, 1) 						# mean 0, variance 1
        GV.rslist[-1].start()
        
        # Define conversion factor so membrane noise is scaled correctly for the section
        self.ka = pi * parameters['diam'] * parameters['L'] * parameters['cm'] * 1e-5 # Conversion factor is the same as in Izh.mod
        
        # Create the membrane noise and associate with the respective random stream
        self.noise = h.InGauss(0.5, sec=self.soma)
        self.noise.mean = self.ka * parameters['noise_mean']	# nA
        self.noise.stdev = self.ka * parameters['noise_stdev']	# nA
        self.noise.dur = 1e9 									# "forever"
        self.noise.noiseFromRandom(GV.rslist[-1].r)
        
        # Add excitatory and inhibitory synapses to the cell, i.e. add to the soma section
        self.esyn = h.ExpSyn(0.5, sec=self.soma)
        self.esyn.e = parameters['e_rev_e']					# Excitatory reversal potential
        self.esyn.tau = parameters['tau_e']					# Excitatory time constant
        
        self.isyn = h.ExpSyn(0.5, sec=self.soma)
        self.isyn.e = parameters['e_rev_i']					# Inhibitory reversal potential
        self.isyn.tau = parameters['tau_i']					# Inhibitory time constant
        
        # needed for PyNN
        self.source_section = self.soma
        self.source = self.soma(0.5)._ref_v
        self.rec = h.NetCon(self.soma(0.5)._ref_v, None,
                            self.get_threshold(), 0.0, 0.0,
                            sec=self.source_section)
        self.spike_times = h.Vector(0)
        self.parameter_names = ('L', 'diam', 'nseg', 'Ra', 'cm', 'a_', 'b', 'c', 'd', 'thresh', 'thresh_noise', 'bias_current_amp', 'noise_mean', 'noise_stdev', 'e_rev_e', 'e_rev_i', 'tau_e', 'tau_i')
        self.traces = {}
        self.recording_time = False

    L = _new_property('soma', 'L')
    diam = _new_property('soma', 'diam')
    nseg = _new_property('soma', 'nseg')
    Ra = _new_property('soma', 'Ra')
    cm = _new_property('soma', 'cm')
    a_ = _new_property('izh', 'a')
    b = _new_property('izh', 'b')
    c = _new_property('izh', 'c')
    d = _new_property('izh', 'd')
    thresh = _new_property('izh', 'thresh')
    bias_current_amp = _new_property('izh', 'amp')
    noise_mean = _new_property('noise', 'mean')
    noise_stdev = _new_property('noise', 'stdev')
    
    @property
    def excitatory(self):
        return self.esyn

    @property
    def inhibitory(self):
        return self.isyn

    def _get_e_e(self):
        return self.esyn.e

    def _set_e_e(self, value):
        self.esyn.e = value
    e_rev_e = property(fget=_get_e_e, fset=_set_e_e)

    def _get_e_i(self):
        return self.isyn.e

    def _set_e_i(self, value):
        self.isyn.e = value
    e_rev_i = property(fget=_get_e_i, fset=_set_e_i)
    
    def _get_tau_e(self):
        return self.esyn.tau

    def _set_tau_e(self, value):
        self.esyn.tau = value
    tau_e = property(fget=_get_tau_e, fset=_set_tau_e)

    def _get_tau_i(self):
        return self.isyn.tau

    def _set_tau_i(self, value):
        self.isyn.tau = value
    tau_i = property(fget=_get_tau_i, fset=_set_tau_i)
    
    def get_threshold(self):
        return self.izh.thresh
    
    def area(self):
        """Membrane area in µm²"""
        return pi * self.soma.L * self.soma.diam
    
    def memb_init(self):
        for seg in self.soma:
            seg.v = self.v_init
        self.izh.u = self.izh.b * self.izh.c
    
class Izhikevich_Type(NativeCellType):
    default_parameters = {'L': 10.0, 'diam': 10.0/pi, 'nseg': 1, 'Ra': 150, 'cm': 1, 'a_': 0.02, 'b': 0.2, 'c': -65.0, 'd': 2.0, 'thresh': 30.0, 'thresh_noise': 0.0, 'bias_current_amp': 0.0, 'noise_mean': 0.0, 'noise_stdev': 0.0, 'e_rev_e': 0.0, 'tau_e': 0.0, 'e_rev_i': 0.0, 'tau_i': 0.0}
    default_initial_values = {'v': -65.0, 'u': 0}
    recordable = ['soma(0.5).v', 'esyn.i', 'isyn.i']
    units = {'soma(0.5).v' : 'mV', 'esyn.i': 'nA', 'isyn.i': 'nA'}    
    receptor_types = ['esyn', 'isyn']
    model = Izhikevich
