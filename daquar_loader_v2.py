import os
import clip
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

class RandomResizedCropAndInterpolation:
    def __init__(self, size, scale=(0.5, 1.0)):
        self.size = size
        self.scale = scale
        
    def __call__(self, img):
        transform = transforms.RandomResizedCrop(
            self.size, 
            scale=self.scale,
            interpolation=random.choice([
                transforms.InterpolationMode.BILINEAR,
                transforms.InterpolationMode.BICUBIC,
                transforms.InterpolationMode.LANCZOS
            ])
        )
        return transform(img)

def get_clip_augmentation(base_transforms):
    """Get enhanced augmentation pipeline for CLIP training"""
    # Augmentations from original CLIP training + enhancements
    return transforms.Compose([
        RandomResizedCropAndInterpolation(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.2),
        transforms.ToTensor(),
        base_transforms[-1]  # Use the final normalization from CLIP's preprocessing
    ])

# Update your dataset class to use this augmentation
class DAQUARDatasetAugmented(Dataset):
    def __init__(self, dataset, image_dir, transform=None, eval_mode=False, eval_transform=None):
        self.dataset = dataset
        self.image_dir = image_dir
        self.transform = transform
        self.eval_mode = eval_mode
        self.eval_transform = eval_transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Load image
        image_id = item['image_id']
        image_path = os.path.join(self.image_dir, f"{image_id}.png")
        image = Image.open(image_path).convert("RGB")
        
        if self.eval_mode and self.eval_transform:
            image = self.eval_transform(image)
        elif self.transform:
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


# Define your dataloaders with augmentation
def create_improved_dataloaders(clip_preprocess, dataset, image_dir, batch_size=64):

    # Create strong augmentation for training
    train_transform = get_clip_augmentation(clip_preprocess.transforms)
    
    # Create datasets
    train_dataset = DAQUARDatasetAugmented(dataset["train"], image_dir, transform=train_transform)
    test_dataset = DAQUARDatasetAugmented(dataset["test"], image_dir, eval_mode=True, eval_transform=clip_preprocess)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2
    )
    
    
    return train_loader, test_loader