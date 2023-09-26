# Tsodyks-Markram Synapse Parameters

# -----------------+-------+---------+-----------+--------+-------+-------+
#   Synapse Type   |   U   | tau_rec | tau_facil |   A    | tau_s | delay |
# -----------------+-------+---------+-----------+--------+-------+-------+
# Exc Facilitating |  0.09 |     138 |       670 |    0.2 |     3 |     1 |
# -----------------+-------+---------+-----------+--------+-------+-------+
# Exc Depressing   |   0.5 |     671 |        17 |   0.63 |     3 |     1 |
# -----------------+-------+---------+-----------+--------+-------+-------+
# Exc Pseudolinear |  0.29 |     329 |       326 |   0.17 |     3 |     1 |
# -----------------+-------+---------+-----------+--------+-------+-------+
# Inh Facilitating | 0.016 |      45 |       376 |   0.08 |    11 |     1 |
# -----------------+-------+---------+-----------+--------+-------+-------+
# Inh Depressing   |  0.25 |     706 |        21 |   0.75 |    11 |     1 |
# -----------------+-------+---------+-----------+--------+-------+-------+
# Inh Pseudolinear |  0.32 |     144 |        62 |   0.17 |    11 |     1 |
# -----------------+-------+---------+-----------+--------+-------+-------+

# Neuron Parameters
# Taken from AmirAli's Matlab model: see the supplementary material!

# +-------------------+------+------+-------+------+----------+--------------+
# |      Neuron       |  a   |  b   |   c   |  d   | v_thresh | Bias current |
# +-------------------+------+------+-------+------+----------+--------------+
# | RS                | 0.02 |  0.2 | -65.0 |  8.0 |     30.0 |       3.5697 |
# | IB                | 0.02 |  0.2 | -55.0 |  4.0 |     30.0 |       3.6689 |
# | FS                |  0.1 |  0.2 | -65.0 |  2.0 |     30.0 |       3.8672 |
# | LTS               | 0.02 | 0.25 | -65.0 |  2.0 |     30.0 |       0.4958 |
# | TC                | 0.02 | 0.25 | -65.0 | 0.05 |     30.0 |       0.6941 |
# | TR                | 0.02 | 0.25 | -65.0 | 2.05 |     30.0 |       0.6941 |
# +-------------------+------+------+-------+------+----------+--------------+

RS_parameters = {
    'a_': 0.02,
    'b': 0.2,
    'c': -65.0,
    'd': 8.0,
    'thresh': 30.0,
    'thresh_noise': 0.5,
    'bias_current_amp': 4,
    'noise_mean': 0,
    'noise_stdev': 1.5,
    'e_rev_e': 0.0,
    'tau_e': 3.0,
    'e_rev_i': -70.0,
    'tau_i': 11.0
    }
IB_parameters = {
    'a_': 0.02,
    'b': 0.2,
    'c': -55.0,
    'd': 4.0,
    'thresh': 30.0,
    'thresh_noise': 0.5,
    'bias_current_amp': 4,
    'noise_mean': 0,
    'noise_stdev': 1.5,
    'e_rev_e': 0.0,
    'tau_e': 3.0,
    'e_rev_i': -70.0,
    'tau_i': 11.0
    }
FS_parameters = {
    'a_': 0.1,
    'b': 0.2,
    'c': -65.0,
    'd': 2.0,
    'thresh': 30.0,
    'thresh_noise': 0.5,
    'bias_current_amp': 4,
    'noise_mean': 0,
    'noise_stdev': 1.5,
    'e_rev_e': 0.0,
    'tau_e': 3.0,
    'e_rev_i': -70.0,
    'tau_i': 11.0
    }
LTS_parameters = {
    'a_': 0.02,
    'b': 0.25,
    'c': -65.0,
    'd': 2.0,
    'thresh': 30.0,
    'thresh_noise': 0.5,
    'bias_current_amp': 1,
    'noise_mean': 0,
    'noise_stdev': 1.5,
    'e_rev_e': 0.0,
    'tau_e': 3.0,
    'e_rev_i': -70.0,
    'tau_i': 11.0
    }
Rel_TC_parameters = {
    'a_': 0.02,
    'b': 0.25,
    'c': -65.0,
    'd': 0.05,
    'thresh': 30.0,
    'thresh_noise': 0.5,
    'bias_current_amp': 1,
    'noise_mean': 0,
    'noise_stdev': 1.5,
    'e_rev_e': 0.0,
    'tau_e': 3.0,
    'e_rev_i': -70.0,
    'tau_i': 11.0
    }
Ret_parameters = {
    'a_': 0.02,
    'b': 0.25,
    'c': -65.0,
    'd': 2.05,
    'thresh': 30.0,
    'thresh_noise': 0.5,
    'bias_current_amp': 1,
    'noise_mean': 0,
    'noise_stdev': 1.5,
    'e_rev_e': 0.0,
    'tau_e': 3.0,
    'e_rev_i': -70.0,
    'tau_i': 11.0
    }

# Maximum coupling weights from row to columns of TCM substructures
# Two sets of weights are given for normal and PD conditions
# Data taken from the Matlab model

# NORMAL
#
# +---------+------+------+-------+------+------+------+
# | FROM\TO |  S   |  M   |   D   |  CI  | TCR  | TRN  |
# +---------+------+------+-------+------+------+------+
# | S       |  -10 |  300 |   300 |  200 |    0 |    0 |
# | M       |   10 |  -10 |     0 |  200 |    0 |    0 |
# | D       |  500 |    0 |   -10 |  200 |  700 |  700 |
# | CI      | -500 | -300 | -7500 | -500 |    0 |    0 |
# | TCR     |    0 |    0 |    10 |   10 |    0 | 1000 |
# | TRN     |    0 |    0 |     0 |    0 | -500 |  -50 |
# +---------+------+------+-------+------+------+------+

# PD
#
# +---------+------+------+-------+------+-------+-----+
# | FROM\TO |  S   |  M   |   D   |  CI  |  TCR  | TRN |
# +---------+------+------+-------+------+-------+-----+
# | S       |  -50 |   10 |   300 |  200 |     0 |   0 |
# | M       |  300 |  -50 |     0 |  200 |     0 |   0 |
# | D       |  500 |    0 |   -50 |  200 |   100 | 100 |
# | CI      | -750 | -750 | -5000 |  -50 |     0 |   0 |
# | TCR     |    0 |    0 |  1000 | 1000 |     0 | 500 |
# | TRN     |    0 |    0 |     0 |    0 | -2500 | -50 |
# +---------+------+------+-------+------+-------+-----+

# Connectivity data format: max_g[TO][FROM]
max_g_pd = {}
max_g_pd["S"] = {
    "S": 5e1,
    "M": 3e2,
    "D": 5e2,
    "CI": 7.5e2,
    "TRN": 0,
    "TCR": 0
}
max_g_pd["M"] = {
    "S": 1e1,
    "M": 5e1,
    "D": 0,
    "CI": 7.5e2,
    "TRN": 0,
    "TCR": 0
}
max_g_pd["D"] = {
    "S": 3e2,
    "M": 0,
    "D": 5e1,
    "CI": 5e3,
    "TRN": 0,
    "TCR": 1e3
}
max_g_pd["CI"] = {
    "S": 2e2,
    "M": 2e2,
    "D": 2e2,
    "CI": 5e1,
    "TRN": 0,
    "TCR": 1e3
}
max_g_pd["TRN"] = {
    "S": 0,
    "M": 0,
    "D": 1e2,
    "CI": 0,
    "TRN": 5e1,
    "TCR": 5e2
}
max_g_pd["TCR"] = {
    "S": 0,
    "M": 0,
    "D": 1e2,
    "CI": 0,
    "TRN": 2.5e3,
    "TCR": 0
}

# Connectivity data format: max_g[TO][FROM]
max_g_normal = {}
max_g_normal["S"] = {
    "S": 1e1,
    "M": 1e2,
    "D": 5e2,
    "CI": 5e2,
    "TRN": 0,
    "TCR": 0
}
max_g_normal["M"] = {
    "S": 3e2,
    "M": 1e1,
    "D": 0,
    "CI": 3e2,
    "TRN": 0,
    "TCR": 0
}
max_g_normal["D"] = {
    "S": 3e2,
    "M": 0,
    "D": 1e1,
    "CI": 7.5e3,
    "TRN": 0,
    "TCR": 1e1
}
max_g_normal["CI"] = {
    "S": 2e2,
    "M": 2e2,
    "D": 2e2,
    "CI": 5e2,
    "TRN": 0,
    "TCR": 1e1
}
max_g_normal["TRN"] = {
    "S": 0,
    "M": 0,
    "D": 7e2,
    "CI": 0,
    "TRN": 5e1,
    "TCR": 1e3
}
max_g_normal["TCR"] = {
    "S": 0,
    "M": 0,
    "D": 7e2,
    "CI": 0,
    "TRN": 5e2,
    "TCR": 0
}
