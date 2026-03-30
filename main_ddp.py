import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from PIL import Image

from mrl_clip_model import MRL_CLIP_VQA
import clip


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

def setup(rank, world_size):
    """
    Initialize the distributed environment.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """
    Clean up the distributed environment.
    """
    dist.destroy_process_group()


def create_daquar_dataloaders_ddp(dataset, image_dir, rank, world_size, batch_size=16):
    """Create DataLoaders optimized for distributed training"""

    clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda:0", jit=False)
    
    # Create datasets
    train_dataset = DAQUARDataset(dataset["train"], image_dir, transform=clip_preprocess)
    test_dataset = DAQUARDataset(dataset["test"], image_dir, transform=clip_preprocess)
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    test_sampler = DistributedSampler(
        test_dataset, 
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create dataloaders with performance optimizations
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=2,  # Increase for better CPU utilization
        pin_memory=True,  # Faster data transfer to GPU
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True  
    )
    
    return train_loader, test_loader, clip_model


def train_mrl_clip_vqa_ddp(rank, world_size, dataset, image_dir, batch_size=32, num_epochs=5):
    """Training function that runs on each GPU"""
    
    # Setup the distributed environment
    setup(rank, world_size)
    
    # Set device
    device = torch.device(f"cuda:{rank}")
    print(f"Running on rank {rank}, device: {device}")
    
    # Create dataloaders and get CLIP model
    train_loader, test_loader, clip_model = create_daquar_dataloaders_ddp(dataset, image_dir, rank, world_size, batch_size)
    
    # Create MRL-CLIP-VQA model
    nesting_list = [16, 32, 64, 128, 256, 512]
    relative_importance = [1] * len(nesting_list)  # Equal weighting initially

    model = MRL_CLIP_VQA(
        clip_model=clip_model,
        nesting_list=nesting_list,
        relative_importance=relative_importance
    ).to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[rank])
    
    # Create optimizer (only optimize the MRL projection)
    optimizer = torch.optim.AdamW(
        model.module.mrl_visual_projection.parameters(),  # Access through .module for DDP
        lr=5e-5,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.1
    )
    
    # Training logs
    train_losses = []
    nested_train_losses = {i: [] for i in range(len(nesting_list))}
    test_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        
        # Training phase
        model.train()
        epoch_loss = 0.0
        nested_losses = [0.0] * len(nesting_list)

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            questions = batch['question'].to(device)  # Ensure questions are on the same device
            
            optimizer.zero_grad()
            loss, per_granularity_losses = model(images, questions)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            for i, gl in enumerate(per_granularity_losses):
                nested_losses[i] += gl.item()
            
            # Print batch progress (only from rank 0)
            if batch_idx % 20 == 0 and rank == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Average losses across all GPUs
        avg_train_loss = epoch_loss / len(train_loader)
        train_loss_tensor = torch.tensor(avg_train_loss).to(device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        avg_train_loss = train_loss_tensor.item() / world_size
        
        # Evaluate
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                questions = batch['question'].to(device)  # Ensure questions are on the same device
                
                loss, _ = model(images, questions)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_loss_tensor = torch.tensor(avg_test_loss).to(device)
        dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
        avg_test_loss = test_loss_tensor.item() / world_size

#         # Save logs only on rank 0
#         if rank == 0:
#             train_losses.append(avg_train_loss)
#             test_losses.append(avg_test_loss)
            
#             # Average nested losses
#             for i in range(len(nested_losses)):
#                 avg_nested_loss = nested_losses[i] / len(train_loader)
#                 nested_loss_tensor = torch.tensor(avg_nested_loss).to(device)
#                 dist.all_reduce(nested_loss_tensor, op=dist.ReduceOp.SUM)
#                 nested_train_losses[i].append(nested_loss_tensor.item() / world_size)
            
#             # Print epoch summary
#             print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")
#             print("Nested losses:")
#             for i, dim in enumerate(nesting_list):
#                 print(f"  Dim {dim}: {nested_train_losses[i][-1]:.4f}")
    
#     # Only save the final model (rank 0 only)
#     if rank == 0:
#         torch.save({
#             'model_state_dict': model.module.state_dict(),  # Save without DDP wrapper
#             'optimizer_state_dict': optimizer.state_dict(),
#             'train_loss': train_losses[-1],
#             'test_loss': test_losses[-1],
#             'nesting_list': nesting_list,
#             'relative_importance': relative_importance
#         }, 'artifacts/mrl_clip_vqa_ddp_final.pth')

    # Clean up
    cleanup()


# def train_ddp_wrapper(dataset, image_dir, nesting_list, relative_importance, num_gpus=2):
#     """Wrapper function to spawn processes for DDP"""
    
#     # Get number of available GPUs
#     world_size = min(torch.cuda.device_count(), num_gpus)
#     if world_size < 1:
#         raise ValueError("No GPUs available for distributed training.")
        
#     # Increase batch size proportionally to number of GPUs
#     batch_size = 32 * world_size
#     print(f"Training with {world_size} GPUs, effective batch size: {batch_size}")
    
#     # Spawn processes
#     args = (world_size, dataset, image_dir, batch_size, 5)  # 5 epochs
#     mp.spawn(train_mrl_clip_vqa_ddp, args=args, nprocs=world_size, join=True)
    
#     # Load final model from rank 0's checkpoint
#     checkpoint = torch.load('mrl_clip_vqa_ddp_final.pth')  # Last epoch
    
#     # Create a non-DDP model to return
#     clip_model, _ = clip.load("ViT-B/32", device="cuda:0", jit=False)
#     clip_model = clip_model.float()  # Ensure model is in float32 mode

#     model = MRL_CLIP_VQA(
#         clip_model=clip_model,
#         nesting_list=nesting_list,
#         relative_importance=relative_importance
#     ).to("cuda:0").float()  
    
#     # Load trained weights
#     model.load_state_dict(checkpoint['model_state_dict'])
    
#     return model

if __name__ == "__main__":
    # Example usage
    from datasets import load_dataset
    
    # Load dataset from CSV files for training and testing
    dataset = load_dataset(
        "csv", 
        data_files={
            "train": "daquar-dataset/data_train.csv",
            "test": "daquar-dataset/data_eval.csv"
        },
        cache_dir="cache"
    )

    # read answer space from file and split into an array by line
    with open("daquar-dataset/answer_space.txt") as f:
        answer_space = f.read().splitlines()

    # label each item in the dataset with their respective answers
    dataset = dataset.map(
        lambda examples: {
            'label': [
                answer_space.index(ans.replace(" ", "").split(",")[0])  # select the 1st answer if multiple answers are provided
                for ans in examples['answer']
            ]
        },
        batched=True
    )
    
    # Image directory
    image_dir = "daquar-dataset/images"
    
    # Nesting list for MRL-CLIP-VQA
    nesting_list = [16, 32, 64, 128, 256, 512]  # Define nesting granularities
    relative_importance = [1] * len(nesting_list)  # Equal weighting initially

    # Set number of GPUs to use
    num_gpus = 2  # Adjust based on your setup

    # Get number of available GPUs
    world_size = min(torch.cuda.device_count(), num_gpus)
    if world_size < 1:
        raise ValueError("No GPUs available for distributed training.")
        
    # Increase batch size proportionally to number of GPUs
    batch_size = 8 * world_size
    print(f"Training with {world_size} GPUs, effective batch size: {batch_size}")
    
    # Spawn processes
    args = (world_size, dataset, image_dir, batch_size, 5)  # 5 epochs
    mp.spawn(train_mrl_clip_vqa_ddp, args=args, nprocs=world_size, join=True)
    
    # Load final model from rank 0's checkpoint
    checkpoint = torch.load('mrl_clip_vqa_ddp_final.pth')  # Last epoch
    
    # Create a non-DDP model to return
    clip_model, _ = clip.load("ViT-B/32", device="cuda:0", jit=False)
    clip_model = clip_model.float()  # Ensure model is in float32 mode

    model = MRL_CLIP_VQA(
        clip_model=clip_model,
        nesting_list=nesting_list,
        relative_importance=relative_importance
    ).to("cuda:0").float()  
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # # Train the model using DDP
    # trained_model = train_ddp_wrapper(dataset, image_dir, nesting_list, relative_importance, num_gpus=2)

    #save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'nesting_list': nesting_list,
        'relative_importance': relative_importance
    }, 'artifacts/mrl_clip_vqa_final_model.pth')
    
    print("Training complete! Model is ready for evaluation.")