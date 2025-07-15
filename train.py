import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import MiniLLM
from tokenizer import SimpleTokenizer

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

def train_model(model, dataloader, num_epochs=10, lr=3e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}: Average Loss = {avg_loss:.4f}')
    
    return losses

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(tokens)
            next_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            
    return tokenizer.decode(tokens[0].cpu().tolist())

if __name__ == "__main__":
    sample_text = """
    The quick brown fox jumps over the lazy dog. The dog was sleeping peacefully under the warm sun.
    In the distance, birds were singing melodious songs. The fox continued its journey through the forest.
    Trees swayed gently in the breeze. Nature was at its finest on this beautiful day.
    """
    
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(sample_text)
    
    dataset = TextDataset(sample_text, tokenizer, seq_len=32)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = MiniLLM(vocab_size=tokenizer.vocab_size, d_model=128, n_heads=8, n_layers=4, d_ff=512)
    
    print("Training model...")
    losses = train_model(model, dataloader, num_epochs=20, lr=1e-3)
    
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()
    
    print("\nGenerating text...")
    generated = generate_text(model, tokenizer, "The quick", max_length=50)
    print(f"Generated text: {generated}")
    
    torch.save(model.state_dict(), 'mini_llm.pth')
    print("Model saved as mini_llm.pth")