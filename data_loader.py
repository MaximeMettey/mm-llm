import requests
import re

def get_simple_dataset():
    """Get a small but diverse dataset for training"""
    
    # Option 1: Simple stories and facts
    simple_texts = [
        "The sun rises in the east and sets in the west. Birds fly south for winter.",
        "Water boils at 100 degrees. Ice melts when it gets warm. Rain falls from clouds.",
        "Cats like to sleep in sunny spots. Dogs wag their tails when happy. Fish swim in water.",
        "Trees grow tall with strong roots. Flowers bloom in spring. Leaves fall in autumn.",
        "People eat food to stay healthy. Sleep helps the body rest. Exercise makes us strong.",
        "Books contain stories and knowledge. Music makes people happy. Art shows beauty.",
        "Cars drive on roads. Planes fly in the sky. Ships sail on water.",
        "The moon shines at night. Stars twinkle in the dark sky. The earth spins around.",
        "Children play games and learn new things. Friends help each other. Family is important.",
        "Time moves forward. Yesterday is past. Tomorrow is future. Today is now.",
    ]
    
    # Option 2: Get public domain text (small sample)
    try:
        # Alice in Wonderland (public domain)
        response = requests.get("https://www.gutenberg.org/files/11/11-0.txt", timeout=10)
        if response.status_code == 200:
            alice_text = response.text
            # Clean and take first 5000 characters
            alice_text = re.sub(r'\r\n', '\n', alice_text)
            alice_text = alice_text[alice_text.find("Alice was beginning"):alice_text.find("Alice was beginning") + 5000]
            simple_texts.append(alice_text)
            print("Added Alice in Wonderland text")
    except:
        print("Couldn't fetch Alice in Wonderland, using local data only")
    
    return "\n".join(simple_texts)

def get_generated_dataset():
    """Generate synthetic training data"""
    
    # Templates for generating varied sentences
    subjects = ["The cat", "A dog", "The bird", "A fish", "The tree", "A flower", "The child", "A person"]
    verbs = ["runs", "jumps", "walks", "sleeps", "plays", "sings", "dances", "thinks"]
    objects = ["in the garden", "near the house", "by the river", "under the tree", "on the grass", "in the sun"]
    
    generated = []
    
    for subj in subjects:
        for verb in verbs:
            for obj in objects[:3]:  # Limit combinations
                generated.append(f"{subj} {verb} {obj}.")
    
    # Add some variety
    for i in range(len(generated)):
        if i % 3 == 0:
            generated[i] = generated[i] + " It was a beautiful day."
        elif i % 3 == 1:
            generated[i] = generated[i] + " The weather was perfect."
    
    return " ".join(generated)

if __name__ == "__main__":
    print("Getting simple dataset...")
    simple_data = get_simple_dataset()
    print(f"Simple dataset length: {len(simple_data)} characters")
    
    print("\nGetting generated dataset...")
    generated_data = get_generated_dataset()
    print(f"Generated dataset length: {len(generated_data)} characters")
    
    # Combine both
    combined = simple_data + "\n" + generated_data
    
    with open('training_data.txt', 'w') as f:
        f.write(combined)
    
    print(f"\nTotal dataset: {len(combined)} characters")
    print("Saved to training_data.txt")