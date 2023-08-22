# -*- coding: utf-8 -*-
"""
Created on Wed April 03 14:27:26 2019

Description: Define arrays for holding global references for cortical network membrane noise

			Paper Ref . . . 
.
			

@author: John Fleming, john.fleming@ucdconnect.ie
"""


# List to append the random streams to (so won't disappear after initialization) for membrane noise
rslist = []

# List to append the random objects to (so won't disappear after initialization) for threshold noise
thresh_rslist = []

# Variable to hold the offset of the random streams
random_stream_offset = 0

# Index variable for generating the independent random stream for each neuron
stream_index = 0
