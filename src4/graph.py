import numpy as np
import matplotlib.pyplot as plt
from synthetic_data import generate_physical_trace_map 
from real_sequence_dataset import RealSequenceDataset  # Import your actual function


def synth_graphs():
    # Parameters
    sequence_len = 10
    range_bins = 64

    # Create subplots for all 5 classes
    fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

    for num_people in range(1, 6):
        # Generate one synthetic sequence
        sequence = generate_physical_trace_map(num_people=num_people, sequence_len=sequence_len, range_bins=range_bins)
        
        # Take first frame
        range_profile = sequence[0]
        
        # Plot
        axs[num_people-1].plot(range_profile)
        axs[num_people-1].set_title(f"{num_people} Person{'s' if num_people > 1 else ''}")
        axs[num_people-1].set_xlabel("Range Bin")
        axs[num_people-1].grid(True, linestyle='--', alpha=0.5)

    axs[0].set_ylabel("Log-Amplitude")
    plt.suptitle("Example Synthetic Range Profiles for Different Crowd Sizes")
    plt.tight_layout()
    plt.show()


def real_graphs():
   # Parameters matching your dataset
    data_path = r"C:/Users/smith/Projects/people-gait/room1"
    sequence_len = 10
    range_bins = 64

    # Load entire dataset WITHOUT shuffling
    dataset = RealSequenceDataset(root_dir=data_path, bins=range_bins, sequence_len=sequence_len, limit_per_label=1, shuffle=False)

    # Group samples by label
    label_to_sample = {}

    for i in range(len(dataset)):
        x, label = dataset[i]
        label = label.item()
        if label not in label_to_sample:
            label_to_sample[label] = x
        if len(label_to_sample) == 5:
            break

    # Sort by label order 1–5
    labels_sorted = sorted(label_to_sample.keys())

    fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharey=True)

    for idx, label in enumerate(labels_sorted):
        x = label_to_sample[label]
        range_profile = x[0].numpy()

        axs[idx].plot(range_profile, linewidth=2)
        axs[idx].set_title(f"{label} Person{'s' if label > 1 else ''}")
        axs[idx].set_xlabel("Range Bin")
        axs[idx].grid(True, linestyle='--', alpha=0.5)
        #axs[idx].set_ylim(0, 2)

    axs[0].set_ylabel("Log-Amplitude")
    plt.suptitle("Example Real Range Profiles from People-Gait")
    plt.tight_layout()
    plt.show()

def heat_map():
   # General parameters
    sequence_len = 10
    range_bins = 64

    # === Synthetic Data ===
    num_people = 3
    synthetic_sequence = generate_physical_trace_map(num_people=num_people, sequence_len=sequence_len, range_bins=range_bins)

    plt.figure(figsize=(8, 5))
    plt.imshow(synthetic_sequence.T, aspect='auto', origin='lower', cmap='jet', interpolation='bicubic')
    plt.colorbar(label='Log-Amplitude')
    plt.title(f"Synthetic Range-Time Heatmap ({num_people} People)")
    plt.xlabel("Time Frame")
    plt.ylabel("Range Bin")
    plt.show()

    # === Real Data ===
    data_path = r"C:/Users/smith/Projects/people-gait/room1"
    dataset = RealSequenceDataset(root_dir=data_path, bins=range_bins, sequence_len=sequence_len, limit_per_label=1, shuffle=False)

    # Grab one real example (e.g., 3 persons)
    for i in range(len(dataset)):
        x, label = dataset[i]
        if label.item() == 3:
            real_sequence = x.numpy()
            break

    plt.figure(figsize=(8, 5))
    plt.imshow(real_sequence.T, aspect='auto', origin='lower', cmap='jet', interpolation='bicubic')
    plt.colorbar(label='Log-Amplitude')
    plt.title(f"Real Range-Time Heatmap (3 Persons)")
    plt.xlabel("Time Frame")
    plt.ylabel("Range Bin")
    plt.show()

if __name__ == "__main__":
    heat_map()
    #synth_graphs()
    #real_graphs()