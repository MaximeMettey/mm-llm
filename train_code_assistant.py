import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import MiniLLM
from tokenizer import SimpleTokenizer
from code_assistant_data import get_combined_code_dataset

class CodeDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=128):  # Longer sequences for code
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tokens = tokenizer.encode(text)
        
    def __len__(self):
        return len(self.tokens) - self.seq_len
        
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        return torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])

def train_code_assistant(num_epochs=40, lr=5e-4):
    """Train a code assistant model"""
    
    print("Loading code assistant dataset...")
    training_data = get_combined_code_dataset()
    
    print(f"Dataset size: {len(training_data)} characters")
    
    # Build vocabulary
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(training_data)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create dataset with longer sequences for code
    dataset = CodeDataset(training_data, tokenizer, seq_len=128)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # Larger model for code understanding
    model = MiniLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=384,  # Larger embedding
        n_heads=12,   # More attention heads
        n_layers=8,   # More layers for complex patterns
        d_ff=1536,    # Larger feed-forward
        max_seq_len=512,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    losses = []
    
    print("Training code assistant model...")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits, loss = model(inputs, targets)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({
                'loss': loss.item(),
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.2e}')
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'code_assistant_checkpoint_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), 'code_assistant_model.pth')
    torch.save(tokenizer, 'code_assistant_tokenizer.pth')
    
    # Plot training curve
    plt.figure(figsize=(12, 6))
    plt.plot(losses)
    plt.title('Code Assistant Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('code_assistant_training_loss.png')
    plt.show()
    
    return model, tokenizer, losses

def generate_code(model, tokenizer, prompt, max_length=200, temperature=0.7):
    """Generate code completion"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(tokens)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-k sampling for better code generation
            top_k = 20
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            
            # Stop at natural code boundaries
            if tokenizer.decode([next_token.item()]) in ['}', ';', '?>', '\n\n']:
                break
    
    return tokenizer.decode(tokens[0].cpu().tolist())

if __name__ == "__main__":
    print("Training Code Assistant LLM")
    print("=" * 50)
    
    model, tokenizer, losses = train_code_assistant(num_epochs=40, lr=5e-4)
    
    print("\nTesting code generation...")
    test_prompts = [
        "function calculateTotal(",
        "class User {",
        "<?php\nfunction getUserById(",
        "const fetchUsers = async () => {",
        "import React, { useState } from 'react';\n\nfunction App() {"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        generated = generate_code(model, tokenizer, prompt, max_length=100, temperature=0.6)
        print(f"Generated: {generated}")
        print("-" * 50)
    
    print("\nModel training complete!")
    print("Files saved:")
    print("- code_assistant_model.pth")
    print("- code_assistant_tokenizer.pth")
    print("- code_assistant_training_loss.png")