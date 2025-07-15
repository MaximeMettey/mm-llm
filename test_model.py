import torch
from model import MiniLLM
from tokenizer import SimpleTokenizer

# Load the trained model
sample_text = """
The quick brown fox jumps over the lazy dog. The dog was sleeping peacefully under the warm sun.
In the distance, birds were singing melodious songs. The fox continued its journey through the forest.
Trees swayed gently in the breeze. Nature was at its finest on this beautiful day.
"""

tokenizer = SimpleTokenizer()
tokenizer.build_vocab(sample_text)

model = MiniLLM(vocab_size=tokenizer.vocab_size, d_model=128, n_heads=8, n_layers=4, d_ff=512)
model.load_state_dict(torch.load('mini_llm.pth'))
model.eval()

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
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

# Test different prompts
prompts = [
    "The quick",
    "The dog",
    "In the",
    "Trees",
    "Nature"
]

print("Testing your trained model:\n")
for prompt in prompts:
    print(f"Prompt: '{prompt}'")
    generated = generate_text(model, tokenizer, prompt, max_length=50, temperature=0.8)
    print(f"Generated: {generated}")
    print("-" * 50)