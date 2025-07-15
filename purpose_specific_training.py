import torch
from model import MiniLLM
from tokenizer import SimpleTokenizer
from torch.utils.data import DataLoader
from train_improved import TextDataset, train_model
import matplotlib.pyplot as plt

def get_chatbot_data():
    """Data for conversational AI"""
    return """
    Hello! How are you today? I'm doing well, thank you for asking.
    What's your favorite color? I really like blue because it's calming.
    Can you help me with math? Of course! I'd be happy to help you.
    What time is it? I don't have access to the current time, but I can help you find it.
    Tell me a joke. Why don't scientists trust atoms? Because they make up everything!
    How's the weather? I can't check the weather, but I hope it's nice where you are.
    Thank you for your help! You're very welcome! I'm glad I could assist you.
    """

def get_code_completion_data():
    """Data for code completion"""
    return """
    def hello_world():
        print("Hello, World!")
        return "success"
    
    for i in range(10):
        print(f"Number: {i}")
        
    if x > 0:
        print("positive")
    else:
        print("negative")
        
    class MyClass:
        def __init__(self):
            self.value = 0
            
        def increment(self):
            self.value += 1
            
    import numpy as np
    arr = np.array([1, 2, 3])
    print(arr.shape)
    
    try:
        result = 10 / 0
    except ZeroDivisionError:
        print("Cannot divide by zero")
    """

def get_storytelling_data():
    """Data for creative writing"""
    return """
    Once upon a time, in a magical forest, there lived a wise old owl.
    The brave knight rode through the dark valley, seeking the lost treasure.
    On a sunny morning, the little girl discovered a secret garden behind her house.
    The dragon slept peacefully in its cave, dreaming of ancient times.
    In the bustling city, people hurried to work while pigeons danced on rooftops.
    The mysterious book glowed softly in the moonlight, waiting to be opened.
    Adventure awaited beyond the misty mountains, where legends came alive.
    """

def get_factual_data():
    """Data for factual Q&A"""
    return """
    The Earth orbits the Sun once every 365.25 days.
    Water freezes at 0 degrees Celsius and boils at 100 degrees Celsius.
    The human heart has four chambers: two atria and two ventricles.
    Photosynthesis is the process by which plants convert sunlight into energy.
    The speed of light is approximately 300,000 kilometers per second.
    DNA contains the genetic instructions for all living organisms.
    The periodic table organizes elements by their atomic number.
    """

def train_purpose_specific_model(purpose="general"):
    """Train a model for specific purpose"""
    
    data_functions = {
        "chatbot": get_chatbot_data,
        "code": get_code_completion_data,
        "story": get_storytelling_data,
        "facts": get_factual_data
    }
    
    if purpose not in data_functions:
        print(f"Available purposes: {list(data_functions.keys())}")
        return
    
    print(f"Training model for: {purpose}")
    
    # Get purpose-specific data
    training_data = data_functions[purpose]()
    
    # Purpose-specific model configurations
    model_configs = {
        "chatbot": {"d_model": 128, "n_heads": 8, "n_layers": 4, "d_ff": 512},
        "code": {"d_model": 256, "n_heads": 8, "n_layers": 6, "d_ff": 1024},  # Larger for code
        "story": {"d_model": 192, "n_heads": 6, "n_layers": 5, "d_ff": 768},  # Medium for creativity
        "facts": {"d_model": 128, "n_heads": 4, "n_layers": 3, "d_ff": 384}   # Smaller for facts
    }
    
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(training_data)
    
    dataset = TextDataset(training_data, tokenizer, seq_len=32)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    config = model_configs[purpose]
    model = MiniLLM(
        vocab_size=tokenizer.vocab_size,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_ff"]
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train with purpose-specific epochs
    epochs = {"chatbot": 30, "code": 50, "story": 40, "facts": 20}
    losses = train_model(model, dataloader, num_epochs=epochs[purpose], lr=1e-3)
    
    # Save model
    torch.save(model.state_dict(), f'{purpose}_model.pth')
    torch.save(tokenizer, f'{purpose}_tokenizer.pth')
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(f'Training Loss - {purpose.capitalize()} Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f'{purpose}_training_loss.png')
    plt.show()
    
    return model, tokenizer

if __name__ == "__main__":
    print("Purpose-specific LLM training")
    print("Choose a purpose:")
    print("1. chatbot - Conversational AI")
    print("2. code - Code completion")
    print("3. story - Creative writing")
    print("4. facts - Factual Q&A")
    
    choice = input("Enter choice (1-4): ")
    purposes = {"1": "chatbot", "2": "code", "3": "story", "4": "facts"}
    
    if choice in purposes:
        model, tokenizer = train_purpose_specific_model(purposes[choice])
        print(f"\nTrained {purposes[choice]} model successfully!")
    else:
        print("Invalid choice")