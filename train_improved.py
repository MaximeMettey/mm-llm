import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import MiniLLM
from tokenizer import SimpleTokenizer
from data_loader import get_simple_dataset, get_generated_dataset

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=64):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tokens = tokenizer.encode(text)
        
    def __len__(self):
        return len(self.tokens) - self.seq_len
        
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        return torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])

def train_model(model, dataloader, num_epochs=15, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    losses = []
    
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
            progress_bar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})
            
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}: Average Loss = {avg_loss:.4f}')
    
    return losses

if __name__ == "__main__":
    print("Loading improved dataset...")
    
    # Get more diverse data
    simple_data = get_simple_dataset()
    generated_data = get_generated_dataset()
    combined_text = simple_data + "\n" + generated_data
    
    print(f"Total training data: {len(combined_text)} characters")
    
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(combined_text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Use smaller sequence length for efficiency
    dataset = TextDataset(combined_text, tokenizer, seq_len=32)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    
    # Smaller but more efficient model
    model = MiniLLM(
        vocab_size=tokenizer.vocab_size, 
        d_model=128, 
        n_heads=8, 
        n_layers=4, 
        d_ff=384,  # Smaller feed-forward
        max_seq_len=256
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("Training model...")
    losses = train_model(model, dataloader, num_epochs=25, lr=1e-3)
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss - Improved Dataset')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('improved_training_loss.png')
    plt.show()
    
    # Save model and tokenizer
    torch.save(model.state_dict(), 'improved_mini_llm.pth')
    torch.save(tokenizer, 'tokenizer.pth')
    print("Model and tokenizer saved!")