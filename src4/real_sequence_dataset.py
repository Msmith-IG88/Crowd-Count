import pandas as pd
import numpy as np
import torch
import os
from collections import defaultdict
from scipy.ndimage import gaussian_filter

class RealSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, bins=64, sequence_len=5, max_range=10, limit_per_label=None, shuffle=True):
        self.samples = []
        self.labels = []
        label_counts = defaultdict(int)
        for label in range(1, 6):
            print(f"for label {label}")
            dir_path = os.path.join(root_dir, str(label))
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if not file.endswith(".csv"):
                        continue
                    if limit_per_label is not None and label_counts[label] >= limit_per_label:
                        break
                    try:
                        df = pd.read_csv(os.path.join(root, file))
                        frames = sorted(df["Frame #"].unique())
                        for i in range(len(frames) - sequence_len + 1):
                            sequence = []
                            for f in frames[i:i + sequence_len]:
                                pts = df[df["Frame #"] == f][["X", "Y"]].to_numpy()
                                distances = np.linalg.norm(pts, axis=1)
                                hist, _ = np.histogram(distances, bins=bins, range=(0, max_range))
                                hist = gaussian_filter(hist, sigma=1.5)  # ADD GAUSSIAN BLUR
                                hist = np.log(1 + hist)  # ADD LOG COMPRESSION
                                sequence.append((hist).astype(float))
                            self.samples.append(np.stack(sequence))
                            self.labels.append(label)
                            label_counts[label] += 1
                            if limit_per_label is not None and label_counts[label] >= limit_per_label:
                                break
                    except Exception as e:
                        print(f"Failed on {file}: {e}")
        # Shuffle to mix labels randomly
        combined = list(zip(self.samples, self.labels))
        if shuffle:
            np.random.shuffle(combined)
        self.samples, self.labels = zip(*combined)
        self.samples = list(self.samples)
        self.labels = list(self.labels)

        # Optional: print label histogram
        from collections import Counter
        print("Loaded label histogram:", Counter(self.labels))

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.int)

    def __len__(self):
        return len(self.samples)

class RealSequenceDataset_old(torch.utils.data.Dataset):
    def __init__(self, root_dir, bins=64, sequence_len=5, max_range=10, limit_per_label=None):
        self.samples = []
        self.labels = []
        label_counts = defaultdict(int)

        for label in range(1, 6):
            dir_path = os.path.join(root_dir, str(label))
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if not file.endswith(".csv"):
                        continue
                    if limit_per_label is not None and label_counts[label] >= limit_per_label:
                        break

                    try:
                        df = pd.read_csv(os.path.join(root, file))
                        frames = sorted(df["Frame #"].unique())
                        for i in range(len(frames) - sequence_len + 1):
                            sequence = []
                            for f in frames[i:i + sequence_len]:
                                pts = df[df["Frame #"] == f][["X", "Y"]].to_numpy()
                                distances = np.linalg.norm(pts, axis=1)
                                hist, _ = np.histogram(distances, bins=bins, range=(0, max_range))
                                sequence.append((hist > 0).astype(float))
                            self.samples.append(np.stack(sequence))
                            self.labels.append(label)
                            label_counts[label] += 1
                            #print(f"Label {label} now has {label_counts[label]} sequences")
                            if limit_per_label is not None and label_counts[label] >= limit_per_label:
                                break
                    except Exception as e:
                        print(f"Failed on {file}: {e}")

        # Shuffle to mix labels randomly
        combined = list(zip(self.samples, self.labels))
        np.random.shuffle(combined)
        self.samples, self.labels = zip(*combined)
        self.samples = list(self.samples)
        self.labels = list(self.labels)

        # Optional: print label histogram
        from collections import Counter
        print("Loaded label histogram:", Counter(self.labels))



    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.int)

    def __len__(self):
        return len(self.samples)