import csv
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


data_path ="D:\\YUVRAJ\\NOVA\\icml_face_data.csv"
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def load_split(path, split):
    """Read the csv and return images and labels forone usage value."""
    images, labels = [],[]
    with open(path, newline="") as f:
              for row in csv.DictReader(f): # reads header, then one dict per row
                     if row["Usage"] != split:
                            continue
                     pixels = np.array(row["pixels"].split(), dtype=np.uint8)
                     images.append(pixels.reshape(48,48))
                     labels.append(int(row["emotion"]))
    return np.stack(images), np.array(labels)
class FaceDataset(Dataset):
       """Wraps the arrays so PyTorch can index them one example at a time."""
       def __init__(self, images, labels):
        self.images = torch.tensor(images, dtype=torch.float32) /255.0 # scale to 0..1
        self.labels = torch.tensor(labels, dtype=torch.long)

       def __len__(self):
        return len(self.labels)

       def __getitem__(self, idx):
       #add a channel dimension: (48, 48) -> (1, 48, 48), the shape CNNs expect
        return self.images[idx].unsqueeze(0), self.labels[idx]

if __name__ =="__main__":
       #check the split sizes first, so we know what the file contains
       with open(data_path, newline="") as f:
              counts = Counter(row["Usage"] for row in csv.DictReader(f))
       print("usage counts:", dict(counts))

       images, labels = load_split(data_path, "Training")
       print("Training rows:", len(labels))

       dataset = FaceDataset(images, labels)
       loader = DataLoader(dataset, batch_size=64, shuffle=True)
       X,Y = next(iter(loader))
       #x,y = next(iter(loader))
       print("Batch image shape:",tuple(x.shape)) #expect (64, 1, 48 ,48)
       print("Batch label shape:",tuple(y.shape)) #expect (64,)
       print("First 5 labels :", [EMOTIONS[i]for i in y[:5].tolist()])