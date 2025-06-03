import os
import clip

from PIL import Image
from torch.utils.data import Dataset, DataLoader

class DAQUARDataset(Dataset):
    def __init__(self, dataset, image_dir, transform=None):
        self.dataset = dataset
        self.image_dir = image_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Load image
        image_id = item['image_id']
        image_path = os.path.join(self.image_dir, f"{image_id}.png")
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # Get question and answer
        question = item['question']
        answer = item['answer']
        label = item['label']
        
        return {
            'image': image,
            'question': question,
            'answer': answer,
            'label': label
        }


def create_daquar_dataloaders(clip_preprocess, dataset, image_dir, batch_size, num_workers=1):
    
    # Create datasets
    train_dataset = DAQUARDataset(dataset["train"], image_dir, transform=clip_preprocess)
    test_dataset = DAQUARDataset(dataset["test"], image_dir, transform=clip_preprocess)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader
