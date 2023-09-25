import matplotlib.pyplot as plt
from pathlib import Path
import neo
import numpy as np
import sys
import scipy.signal as signal


def plot_population_spikes(filename):
    spike_bin_width_ms = 20
    t_min = None
    t_max = None
    result_file = Path(filename)
    population_results = neo.NeoMatlabIO(result_file).read()
    _, axs = plt.subplots(2, 1, figsize=(15, 10))
    colormap = ["#1E212B", "#FF8427"]
    last_population_name = ""
    colormap_i = -1
    master_spike_list = []
    for i, train in enumerate(population_results[0].segments[0].spiketrains):
        if t_min is None and t_max is None:
            t_min = train.t_start
            t_max = train.t_stop
        if train.annotations["source_population"] != last_population_name:
            colormap_i = (colormap_i + 1) % len(colormap)
            last_population_name = train.annotations["source_population"]
        axs[0].plot(train, np.ones(len(train)) * i,
                    '.', markersize=2, color=colormap[colormap_i])
        axs[0].set_xlabel("Time [ms]")
        axs[0].set_ylabel("neuron number")
        axs[0].set_title(f"{result_file.stem} ({result_file.parent})")
        master_spike_list.append(train.as_array())
    time_bins = np.arange(float(t_min), float(t_max), spike_bin_width_ms)
    binned_spikes = np.zeros(time_bins.shape)
    for i, start in enumerate(time_bins):
        if i == len(time_bins) - 1:
            end = float(t_max) + spike_bin_width_ms
        else:
            end = time_bins[i + 1]
        binned_spikes[i] = np.sum([
            np.count_nonzero(np.bitwise_and(start <= st, st < end))
            for st in master_spike_list])
    axs[1].plot(time_bins, binned_spikes)
    axs[1].set_xlabel("Time [ms]")
    axs[1].set_ylabel(f"Spikes / {spike_bin_width_ms} ms")


def plot_d_layer_lfp(d_layer_file, ci_file):
    d_layer_results = neo.NeoMatlabIO(Path(d_layer_file)).read()
    ci_results = neo.NeoMatlabIO(Path(ci_file)).read()
    sigma = 0.27
    r = 100 * 1e-6
    for s in d_layer_results[0].segments[0].analogsignals:
        if s.name == "esyn.i":
            epsc = s
            break
    for s in ci_results[0].segments[0].analogsignals:
        if s.name == "isyn.i":
            ipsc = s
            break
    lfp = (np.mean(epsc, axis=-1) + np.mean(ipsc, axis=-1))
    lfp /= (4 * np.pi * sigma * r)

    if epsc.sampling_rate == ipsc.sampling_rate:
        fs = epsc.sampling_rate * 1000
    else:
        fs = 1

    fxx, psd = signal.welch(lfp, fs, nperseg=10000)

    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    axs[0].plot(lfp / 1000)
    axs[0].set_xlabel("Samples")
    axs[0].set_ylabel("LFP [mV?]")
    axs[0].set_title("D Layer LFP")

    axs[1].plot(fxx, psd)
    axs[1].set_title("Power spectrum of D layer LFP")
    axs[1].set_xlabel("Frequency [Hz]")
    axs[1].set_ylabel("PSD [mV? ** 2 / Hz]")
    axs[1].set_xlim([0, 100])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_simulation_result.py dirname")
        exit(1)
    result_dirname = Path(sys.argv[1])
    if not result_dirname.exists():
        print("Specify an existing directory!")
        exit(1)

    print(f"Plotting results from {result_dirname}")

    plot_population_spikes(result_dirname / "D_Layer.mat")
    plot_population_spikes(result_dirname / "M_Layer.mat")
    plot_population_spikes(result_dirname / "S_Layer.mat")
    plot_population_spikes(result_dirname / "CI_Neurons.mat")
    plot_population_spikes(result_dirname / "TCR_Nucleus.mat")
    plot_population_spikes(result_dirname / "TR_Nucleus.mat")

    plot_d_layer_lfp(result_dirname / "D_Layer.mat",
                     result_dirname / "CI_Neurons.mat")

    plt.show()
