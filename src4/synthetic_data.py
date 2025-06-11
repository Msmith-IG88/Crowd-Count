import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

def detect_peaks(range_profile, height=0.2, distance=3):
    peaks, _ = find_peaks(range_profile, height=height, distance=distance)
    return len(peaks)

def generate_physical_trace_map(num_people, sequence_len=10, range_bins=64, max_range=10):
    sequence = np.zeros((sequence_len, range_bins))
    positions = np.random.uniform(1, max_range, size=(num_people, 2))  # (x,y) positions
    
    # Physics-based movement parameters
    velocities = np.random.uniform(-0.3, 0.3, size=(num_people, 2))
    
    for t in range(sequence_len):
        # Update positions with physics
        positions += velocities
        
        # Boundary reflection
        positions = np.clip(positions, 0.5, max_range-0.5)
        over_boundary = (positions < 0.5) | (positions > max_range-0.5)
        velocities[over_boundary] *= -1  # Reverse direction at boundaries
        
        # Calculate distances and angles
        distances = np.linalg.norm(positions, axis=1)
        angles = np.arctan2(positions[:, 1], positions[:, 0])
        
        # Sort by distance (closest first)
        sorted_idx = np.argsort(distances)
        visible = [True] * num_people
        
        # Apply occlusion model (paper's crowd shadowing)
        for i in sorted_idx:
            for j in sorted_idx:
                if i == j or not visible[j] or distances[j] > distances[i]:
                    continue
                # Angular separation check
                ang_diff = np.abs(angles[i] - angles[j])
                if ang_diff < (0.3 / distances[j]):  # Simplified occlusion model
                    visible[i] = False
                    break
        
        # Create range profile with sensor noise
        range_profile = np.zeros(range_bins)
        for i in range(num_people):
            if visible[i]:
                bin_idx = int(distances[i] / max_range * range_bins)
                bin_idx = np.clip(bin_idx, 0, range_bins-1)
                
                # Sensor response (Gaussian spread)
                response = np.zeros(range_bins)
                response[bin_idx] = 1.0
                response = gaussian_filter(response, sigma=1.5)
                range_profile += response
        
        # Add environmental noise (multipath, sensor noise)
        noise_level = 0.1 + (num_people / 10)  # More people -> more noise
        range_profile += np.random.normal(0, noise_level, range_bins)
        
        # Apply threshold to create binary map
        threshold = 0.4 + (0.1 * num_people)  # Adaptive threshold
        #range_profile = (range_profile > threshold).astype(float)
        range_profile = gaussian_filter(range_profile, sigma=1.5)
        range_profile = np.clip(range_profile, 0, None)
        range_profile = np.log1p(range_profile)  # log compression

        
        sequence[t] = range_profile
    
    return sequence

class ResidualDataset_old(torch.utils.data.Dataset):
    def __init__(self, num_per_label=500, sequence_len=10, range_bins=64):
        self.samples = []
        self.residuals = []
        self.true_counts = []
        for num_people in range(1, 6):
            for _ in range(num_per_label):
                seq = generate_physical_trace_map(num_people, sequence_len, range_bins)
                # Compute baseline using peak detection
                baselines = [detect_peaks(frame) for frame in seq]
                baseline = np.mean(baselines)
                visible_count = seq.sum(axis=1).mean()
                residual = (num_people - baseline)
                self.samples.append(seq)
                self.true_counts.append(num_people)
                self.residuals.append(residual)

    def __getitem__(self, idx):
        return (
                torch.tensor(self.samples[idx], dtype=torch.float32),
                torch.tensor(self.true_counts[idx], dtype=torch.float32),
                torch.tensor(self.residuals[idx], dtype=torch.float32)
        )


    def __len__(self):
        return len(self.samples)

class ResidualDataset(torch.utils.data.Dataset):
    def __init__(self, num_per_label=500, sequence_len=10, range_bins=64):
        self.samples = []
        self.residuals = []
        self.true_counts = []

        # Handle dict or int input
        if isinstance(num_per_label, int):
            label_counts = {label: num_per_label for label in range(1, 6)}
        elif isinstance(num_per_label, dict):
            label_counts = num_per_label
        else:
            raise ValueError("num_per_label must be an int or a dict")

        # Generate samples
        for num_people in range(1, 6):
            for _ in range(label_counts[num_people]):
                seq = generate_physical_trace_map(num_people, sequence_len, range_bins)
                baselines = [detect_peaks(frame) for frame in seq]
                baseline = np.mean(baselines)
                residual = num_people - baseline

                self.samples.append(seq)
                self.true_counts.append(num_people)
                self.residuals.append(residual)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.samples[idx], dtype=torch.float32),
            torch.tensor(self.true_counts[idx], dtype=torch.float32),
            torch.tensor(self.residuals[idx], dtype=torch.float32)
        )

    def __len__(self):
        return len(self.samples)
