import torch
from tqdm.auto import tqdm

def train_mrl_clip_vqa(model, train_loader, test_loader, optimizer, device, num_epochs=5):
    # Training logs
    train_losses = []
    nested_train_losses = {i: [] for i in range(len(model.nesting_list))}
    test_losses = []
    
    # Move model to device
    model = model.to(device)

    # Create a scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        nested_losses = [0.0] * len(model.nesting_list)
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images = batch['image'].to(device)
            questions = batch['question']
            
            optimizer.zero_grad()
            loss, per_granularity_losses = model(images, questions)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            for i, gl in enumerate(per_granularity_losses):
                nested_losses[i] += gl.item()
            
            # Print batch progress
            if batch_idx % 50 == 0:
                print(f"<< Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f} >>")
        
        # Update learning rate
        scheduler.step()

        # Calculate average training loss
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Average nested losses
        for i in range(len(nested_losses)):
            nested_train_losses[i].append(nested_losses[i] / len(train_loader))
        
        # # Evaluation phase
        # model.eval()
        # test_loss = 0.0
        # with torch.no_grad():
        #     for batch in tqdm(test_loader, desc="Evaluating"):
        #         images = batch['image'].to(device)
        #         questions = batch['question']
                
        #         loss, _ = model(images, questions)
        #         test_loss += loss.item()
        
        # # Calculate average test loss
        # avg_test_loss = test_loss / len(test_loader)
        # test_losses.append(avg_test_loss)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
        
        # Print nested losses
        print("Nested losses:")
        for i, dim in enumerate(model.nesting_list):
            print(f"  Dim {dim}: {nested_train_losses[i][-1]:.4f}")
        
        # # Save model checkpoint
        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'train_loss': avg_train_loss,
        #     'test_loss': avg_test_loss,
        # }, f'artifacts/mrl_clip_vqa_epoch_{epoch}.pth')
    
    history = {
        'train_losses': train_losses,
        # 'test_losses': test_losses,
        'nested_train_losses': nested_train_losses
    }
    return model, history