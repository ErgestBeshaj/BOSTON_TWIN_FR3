import os
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera
from src.classes.BostonTwin import BostonTwin

# Set up environment
gpu_num = 1
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Configure TensorFlow to allocate only as much memory as needed
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

# Random seed for reproducibility
tf.random.set_seed(1)

# Load the BostonTwin dataset
dataset_dir = Path("bostontwin")
bostwin = BostonTwin(dataset_dir)

# Select one scene (e.g., BOS_G_5)
scene_name = "BOS_G_5"
sionna_scene, scene_antennas = bostwin.load_bostontwin(scene_name)

# Print the number of antennas available in the scene
n_antennas = scene_antennas.shape[0]
print(f"There are {n_antennas} antennas in the {scene_name} scene.")

# Antenna array configuration
def configure_scene_arrays(scene):
    scene.tx_array = PlanarArray(num_rows=1, 
                                 num_cols=1, 
                                 vertical_spacing=0.5, 
                                 horizontal_spacing=0.5, 
                                 pattern="tr38901", 
                                 polarization="V")
    
    scene.rx_array = PlanarArray(num_rows=1, 
                                 num_cols=1, 
                                 vertical_spacing=0.5, 
                                 horizontal_spacing=0.5, 
                                 pattern="dipole", 
                                 polarization="cross")


def plot_scene(sionna_scene, tx_ids, rx_ids):
    bostwin.plot_twin(basemap=False)
    plt.title("2D Map with Buildings and Antennas")
    plt.show()

# Compute paths and measure time
def measure_compute_paths(scene):
    start_time = time.time()
    paths = scene.compute_paths(
                    max_depth=2,
                    num_samples=1e5,
                    reflection=True,
                    diffraction=True,
                    scattering=True,
                   )
    
    end_time = time.time()
    compute_time = end_time - start_time
    return paths, compute_time


def compute_ray_type_power(paths):
    # Get unique ray types (LoS, reflected, diffracted, scattered)
    ray_types = np.unique(paths.types.numpy())
    ray_types_dict = {}
    total_power = 0

    # Loop through each ray type
    for ray_type in ray_types:
        # Select paths that belong to the current ray type
        rays_ids = paths.types.numpy() == ray_type

        # Calculate the power for the current ray type by summing path amplitudes
        power_sum = np.abs(np.sum((paths.a.numpy())[:,:,:,:,:, rays_ids[0], :]))**2
        rays_pow = 10 * np.log10(power_sum) if power_sum > 0 else -np.inf
 
        ray_types_dict[ray_type] = rays_pow
        if power_sum > 0:
            total_power += power_sum

    # Convert total power to dB scale
    total_power_db = 10 * np.log10(total_power) if total_power > 0 else -np.inf

    print("Aggregated power by ray type (dB):", ray_types_dict)
    print(f"Total received power (dB): {total_power_db}")



# Case 1: All antennas as transmitters, one receiver at (0, 0, 400e3)
def case1_all_tx_one_rx(receiver_position, num_tx=None):
    # Configure antenna arrays
    configure_scene_arrays(sionna_scene)

    # Remove existing transmitters and receivers
    [sionna_scene.remove(rx) for rx in sionna_scene.receivers]
    [sionna_scene.remove(tx) for tx in sionna_scene.transmitters]

    num_tx = num_tx or n_antennas # - 1  # One antenna is left for the receiver - maybe is not needed
    if num_tx > n_antennas:
        raise ValueError(f"Number of transmitters (num_tx) cannot be greater than or equal to {n_antennas}.")

    # Get all antenna IDs and select random ones for transmitters
    ant_ids = np.arange(n_antennas)
    tx_ids = np.sort(np.random.choice(ant_ids, size=num_tx, replace=False))  # Randomly choose transmitters

    # Add the transmitters 
    bostwin.add_scene_antennas(tx_ids, []) 

    receiver = Receiver(name="receiver_1", position=receiver_position) 
    sionna_scene.add(receiver)
    # print(f"Transmitter IDs: {tx_ids}\nReceiver Position: {receiver_position}")

    # Compute paths and measure time
    paths, compute_time = measure_compute_paths(sionna_scene)
    compute_ray_type_power(paths)

    # plot_scene(sionna_scene, tx_ids, [receiver])
    print(f"Time to compute paths: {compute_time} seconds")


# Case 2: All antennas as receivers, one transmitter at (0, 0, 400e3)
def case2_all_rx_one_tx(transmitter_position, num_rx=None):
    # Configure antenna arrays
    configure_scene_arrays(sionna_scene)

    # Remove existing transmitters and receivers
    [sionna_scene.remove(rx) for rx in sionna_scene.receivers]
    [sionna_scene.remove(tx) for tx in sionna_scene.transmitters]

    # Set the number of receivers, ensuring it's less than the total number of antennas
    num_rx = num_rx or n_antennas # - 1  # at least one antenna is left for the transmitter - maybe we don't need this
    if num_rx > n_antennas:
        raise ValueError(f"Number of receivers (num_rx) cannot be greater than or equal to {n_antennas}.")

    # Get all antenna IDs and select random ones for receivers
    ant_ids = np.arange(n_antennas)
    rx_ids = np.sort(np.random.choice(ant_ids, size=num_rx, replace=False))  # Randomly choose receivers

    # Add the receivers
    bostwin.add_scene_antennas([], rx_ids)  

    transmitter = Transmitter(name="transmitter_1", position=transmitter_position)  
    sionna_scene.add(transmitter)
    # print(f"Receiver IDs: {rx_ids}\nTransmitter Position: {transmitter_position}")

    paths, compute_time = measure_compute_paths(sionna_scene)
    # plot_scene(sionna_scene, [transmitter], rx_ids)
    print(f"Time to compute paths: {compute_time} seconds")

num_tx = int(input(f"Enter the number of transmitters / receivers that want to use (1 to {n_antennas}): "))
num_rx = num_tx # int(input(f"Enter the number of receivers (1 to {n_antennas}): "))

# Set receiver at (0, 0, 400e3)
receiver_position = [0, 0, 400e3]
case1_all_tx_one_rx(receiver_position, num_tx=num_tx)

# Set transmitter at (0, 0, 400e3)
transmitter_position = [0, 0, 400e3]
case2_all_rx_one_tx(transmitter_position, num_rx=num_rx)

