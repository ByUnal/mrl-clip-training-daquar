import clip
import torch
import torch.nn.functional as F

from datasets import load_dataset
from tqdm import tqdm

from daquar_loader import create_daquar_dataloaders
from mrl_clip_model import MRL_CLIP_VQA
from evaluate_vqa import analyze_granularity_performance


# def evaluate_nested_representations(model, dataloader, device, k_values=[1, 5]):
#     """
#     Evaluate the quality of nested representations using retrieval metrics.
    
#     Args:
#         model: Trained MRL-CLIP model
#         dataloader: Evaluation dataloader
#         device: Computation device
#         k_values: List of k values for Recall@k metrics
        
#     Returns:
#         Dictionary of metrics at each granularity level
#     """
#     model.eval()
    
#     # Initialize metrics
#     metrics = {dim: {f"R@{k}": 0.0 for k in k_values} for dim in model.nesting_list}
#     total_samples = 0
    
#     # Store all embeddings for analysis
#     all_img_embeds = {dim: [] for dim in model.nesting_list}
#     all_text_embeds = []
#     all_questions = []
    
#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Evaluating representations"):
#             images = batch['image'].to(device)
#             questions = batch['question']
            
#             # Get nested image embeddings and text embeddings
#             image_embeds = model.encode_image(images)
            
#             question_tokens = clip.tokenize(questions, truncate=True).to(device)
#             text_embeds = model.encode_text(question_tokens)
            
#             # Compute similarity for each granularity
#             batch_size = images.shape[0]
#             total_samples += batch_size
            
#             # Store embeddings for later analysis
#             for i, dim in enumerate(model.nesting_list):
#                 all_img_embeds[dim].append(image_embeds[i].cpu())
#             all_text_embeds.append(text_embeds.cpu())
#             all_questions.extend(questions)
            
#             # Calculate metrics for each granularity
#             for i, dim in enumerate(model.nesting_list):
#                 similarities = image_embeds[i] @ text_embeds.T
                
#                 # For each image, get the top k text matches
#                 _, indices = similarities.topk(max(k_values), dim=1)
                
#                 # Check if the ground truth index (diagonal) is in the top k
#                 ground_truth = torch.arange(batch_size, device=device)
                
#                 for k in k_values:
#                     # Check if ground truth is in top k
#                     correct = indices[:, :k].eq(ground_truth.unsqueeze(1)).any(dim=1)
#                     metrics[dim][f"R@{k}"] += correct.sum().item()
    
#     # Normalize metrics
#     for dim in metrics:
#         for k in k_values:
#             metrics[dim][f"R@{k}"] = metrics[dim][f"R@{k}"] / total_samples * 100.0
    
#     # Concatenate all embeddings
#     for dim in all_img_embeds:
#         all_img_embeds[dim] = torch.cat(all_img_embeds[dim], dim=0)
#     all_text_embeds = torch.cat(all_text_embeds, dim=0)
    
#     # Calculate interdimensional cosine similarity to measure information preservation
#     info_preservation = {}
#     highest_dim = max(model.nesting_list)
#     highest_embeds = all_img_embeds[highest_dim]
    
#     for dim in model.nesting_list:
#         if dim != highest_dim:
#             # Calculate how much information from highest dim is preserved in lower dims
#             lower_embeds = all_img_embeds[dim]
#             # We need to project the lower-dim embeddings to the same space first
#             proj_matrix = torch.zeros(highest_dim, dim).to(device)
#             proj_matrix[:dim, :dim] = torch.eye(dim)
            
#             # Project embeddings to higher dim (with zeros in extra dimensions)
#             projected_lower = F.normalize(F.pad(lower_embeds, (0, highest_dim - dim)), dim=1)
            
#             # Calculate cosine similarity
#             cos_sim = F.cosine_similarity(projected_lower, highest_embeds, dim=1).mean().item()
#             info_preservation[dim] = cos_sim
    
#     return metrics, info_preservation, (all_img_embeds, all_text_embeds, all_questions)

def evaluate_granularity_performance(model, test_loader, device):
    
    # Run analysis
    granularity_results, interesting_questions = analyze_granularity_performance(model, test_loader, device)
    
    # Print performance by granularity
    print("\nPerformance by granularity:")
    for dim, result in granularity_results.items():
        print(f"Dimension {dim}: {result['accuracy'] * 100:.2f}% accuracy")
    
    # Print some interesting examples
    print("\nQuestions with varying performance across granularities:")
    for i, q in enumerate(interesting_questions[:5]):
        print(f"{i+1}. \"{q['question']}\" - Correct at dimensions: {q['correct_at']}")
    
    # # Also visualize representations
    # print("\nCreating visualizations of nested representations...")
    # all_embeddings, all_questions = visualize_nested_representations(model, test_loader, device)
    
    return granularity_results, interesting_questions

if __name__ == "__main__":
    # load dataset from CSV files for training and testing
    dataset = load_dataset(
        "csv", 
        data_files={
            "train": "daquar-dataset/data_train.csv",
            "test": "daquar-dataset/data_eval.csv"
        }
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

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    image_dir = "daquar-dataset/images"

    # Load the checkpoint
    checkpoint_path = "artifacts/mrl_clip_vqa_final_model.pth"  
    checkpoint = torch.load(checkpoint_path, map_location=device)

    batch_size = 64 
    model_name = "ViT-B/32"
    clip_model, clip_preprocess = clip.load(model_name, device="cpu")

    _, test_loader = create_daquar_dataloaders(clip_preprocess, dataset, image_dir, batch_size)

    # Get model configuration from checkpoint
    nesting_list = checkpoint.get('nesting_list')
    relative_importance = checkpoint.get('relative_importance')
    
    # Create MRL-CLIP-VQA model with same architecture
    model = MRL_CLIP_VQA(
        clip_model=clip_model.to(device),
        nesting_list=nesting_list,
        relative_importance=relative_importance
    )
    
    # Load the saved state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print("Model loaded successfully!")
    print(f"Model architecture: {nesting_list} dimensions")
    
    # Load and analyze
    results, interesting_questions = evaluate_granularity_performance(model, test_loader, device)
    print("\n------------------------------------------------\n")
    # metrics, info_preservation, embeddings_data = evaluate_nested_representations(model, test_loader, device)
    
    # # Print metrics
    # for dim in metrics:
    #     print(f"Dimension {dim}:")
    #     for metric_name, value in metrics[dim].items():
    #         print(f"  {metric_name}: {value:.2f}%")
    
    # print("\nInformation preservation:")
    # for dim, value in info_preservation.items():
    #     print(f"  Dim {dim}: {value:.4f}")