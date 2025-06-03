import clip
import torch
from datasets import load_dataset

from daquar_loader import create_daquar_dataloaders
from daquar_loader_v2 import create_improved_dataloaders
from evaluate_vqa import analyze_granularity_performance
from mrl_clip_model import MRL_CLIP_VQA
from train_vqa_model import train_mrl_clip_vqa


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

    # print(dataset)
    print("Dataset loaded and answers labeled.")

    # Image directory
    image_dir = "daquar-dataset/images"

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    MODEL_NAME = "ViT-B/32"
    # Load the CLIP model
    clip_model, clip_preprocess = clip.load(MODEL_NAME, device="cpu")

    # Create dataloaders and get CLIP model
    batch_size = 64  # Adjust based on your GPU memory
    # train_loader, test_loader = create_daquar_dataloaders(clip_preprocess, dataset, image_dir, batch_size)
    train_loader, test_loader = create_improved_dataloaders(clip_preprocess, dataset, image_dir, batch_size)
    print("Dataset is ready.")

    # Move CLIP model to device
    clip_model = clip_model.to(device)

    # Create MRL-CLIP-VQA model
    nesting_list = [16, 32, 64, 128, 256, 512]  # Define nesting granularities
    relative_importance = [1] * len(nesting_list)  # Equal weighting initially

    model = MRL_CLIP_VQA(
        clip_model=clip_model,
        nesting_list=nesting_list,
        relative_importance=relative_importance
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.2
    )
    
    print(f"{MODEL_NAME} is fine-tuned...")
    print("----------\n")
    # Train the model
    trained_model, training_history = train_mrl_clip_vqa(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=10  # Start with a few epochs
    )

    # Save the trained model
    torch.save({
        'model_name': MODEL_NAME,
        'model_state_dict': trained_model.state_dict(),
        'nesting_list': nesting_list,
        'relative_importance': relative_importance
    }, 'artifacts/mrl_clip_vqa_final_model.pth')
    print("Model trained and saved successfully!")

    # Analyze granularity performance
    print("Analyzing granularity performance...")
    granularity_results, interesting_questions = analyze_granularity_performance(trained_model, test_loader, device)

    # Print performance by granularity
    print("Performance by granularity:")
    for dim, result in granularity_results.items():
        print(f"Dimension {dim}: {result['accuracy'] * 100:.2f}% accuracy")

    # Print some interesting examples
    print("\nQuestions with varying performance across granularities:")
    for i, q in enumerate(interesting_questions[:5]):
        print(f"{i+1}. \"{q['question']}\" - Correct at dimensions: {q['correct_at']}")

