import clip
import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def visualize_nested_representations(model, dataloader, device, num_samples=100):
    """
    Visualize t-SNE projections of the nested representations at different granularity levels
    """
    model.eval()
    
    # Collect samples
    all_images = []
    all_questions = []
    all_embeddings = [[] for _ in range(len(model.nesting_list))]
    count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if count >= num_samples:
                break
                
            images = batch['image'].to(device)
            questions = batch['question']
            
            # Get nested embeddings
            nested_embeds = model.encode_image(images)
            
            # Store embeddings
            batch_size = min(images.shape[0], num_samples - count)
            for i, embeds in enumerate(nested_embeds):
                all_embeddings[i].append(embeds[:batch_size].cpu().numpy())
            
            # Store questions for reference
            all_questions.extend(questions[:batch_size])
            count += batch_size
    
    # Concatenate all embeddings
    for i in range(len(all_embeddings)):
        all_embeddings[i] = np.vstack(all_embeddings[i])
    
    # Create visualization
    fig, axes = plt.subplots(1, len(model.nesting_list), figsize=(20, 5))
    
    for i, dim in enumerate(model.nesting_list):
        # Use t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(all_embeddings[i])
        
        # Plot results
        axes[i].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=10)
        axes[i].set_title(f"Dimension: {dim}")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    plt.tight_layout()
    plt.savefig('nested_representations.png')
    plt.show()
    
    return all_embeddings, all_questions

def analyze_granularity_performance(model, dataloader, device):
    """
    Analyze performance at different granularity levels
    """
    model.eval()
    results = {dim: {"correct": 0, "total": 0} for dim in model.nesting_list}
    question_difficulty = {}  # Will store which granularity works best for each question
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing granularity performance"):
            images = batch['image'].to(device)
            questions = batch['question']
            
            # Tokenize questions
            question_tokens = clip.tokenize(questions, truncate=True).to(device)
            
            # Get nested image embeddings
            image_embeds = model.encode_image(images)
            
            # Get text embeddings
            text_embeds = model.encode_text(question_tokens)
            
            # For each granularity level
            for i, dim in enumerate(model.nesting_list):
                # Calculate similarities
                similarities = image_embeds[i] @ text_embeds.T
                
                # Get predictions (highest similarity score)
                _, predictions = similarities.max(dim=1)
                
                # Ground truth (diagonal indices)
                ground_truth = torch.arange(images.shape[0], device=device)
                
                # Compare predictions with ground truth
                correct = (predictions == ground_truth)
                
                # Update stats
                results[dim]["correct"] += correct.sum().item()
                results[dim]["total"] += len(correct)
                
                # Store which questions were answered correctly at this granularity
                for q_idx, (q, is_correct) in enumerate(zip(questions, correct)):
                    if q not in question_difficulty:
                        question_difficulty[q] = []
                    if is_correct:
                        question_difficulty[q].append(dim)
    
    # Calculate accuracy for each granularity level
    for dim in results:
        results[dim]["accuracy"] = results[dim]["correct"] / results[dim]["total"]
    
    # Find questions that are handled differently by different granularities
    interesting_questions = []
    for q, correct_dims in question_difficulty.items():
        if len(correct_dims) > 0 and len(correct_dims) < len(model.nesting_list):
            interesting_questions.append({
                "question": q,
                "correct_at": correct_dims
            })
    
    return results, interesting_questions