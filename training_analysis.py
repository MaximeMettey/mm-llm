import torch
import matplotlib.pyplot as plt
import numpy as np

def analyze_training_progress():
    """Analyze when to add more data vs more training"""
    
    print("Training Progress Analysis")
    print("=" * 50)
    
    # Simulate different scenarios
    epochs = np.arange(1, 101)
    
    # Small dataset scenarios
    small_data_loss = 2.0 * np.exp(-epochs/10) + 0.5  # Plateaus at 0.5
    small_data_overfitting = 2.0 * np.exp(-epochs/8) + 0.3 + 0.1 * np.sin(epochs/5)  # Starts overfitting
    
    # Large dataset scenarios  
    large_data_loss = 2.0 * np.exp(-epochs/20) + 0.2  # Continues improving
    large_data_optimal = 2.0 * np.exp(-epochs/15) + 0.15  # Better final performance
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Small vs Large Dataset
    plt.subplot(1, 3, 1)
    plt.plot(epochs, small_data_loss, 'r-', label='Small Dataset', linewidth=2)
    plt.plot(epochs, large_data_loss, 'b-', label='Large Dataset', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Small vs Large Dataset')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Overfitting Detection
    plt.subplot(1, 3, 2)
    plt.plot(epochs, small_data_overfitting, 'r-', label='Overfitting (Small Data)', linewidth=2)
    plt.plot(epochs, large_data_optimal, 'g-', label='Healthy Training (Large Data)', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Overfitting vs Healthy Training')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Performance Comparison
    plt.subplot(1, 3, 3)
    performance_small = 1 / (1 + small_data_loss)  # Convert loss to performance
    performance_large = 1 / (1 + large_data_loss)
    
    plt.plot(epochs, performance_small, 'r-', label='Small Dataset Performance', linewidth=2)
    plt.plot(epochs, performance_large, 'b-', label='Large Dataset Performance', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Performance Score')
    plt.title('Final Performance Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'small_data_final_loss': small_data_loss[-1],
        'large_data_final_loss': large_data_loss[-1],
        'improvement_ratio': small_data_loss[-1] / large_data_loss[-1]
    }

def get_data_expansion_strategies():
    """Different strategies to expand training data"""
    
    strategies = {
        'web_scraping': {
            'description': 'Scrape GitHub repositories for real code',
            'pros': ['Real-world code', 'Large volume', 'Current practices'],
            'cons': ['License issues', 'Quality varies', 'Need cleaning'],
            'effort': 'Medium',
            'data_multiplier': '50-100x'
        },
        
        'synthetic_generation': {
            'description': 'Generate code using templates and patterns',
            'pros': ['Controlled quality', 'Specific patterns', 'No legal issues'],
            'cons': ['Less diversity', 'May be too regular', 'Limited creativity'],
            'effort': 'Low',
            'data_multiplier': '10-20x'
        },
        
        'documentation_mining': {
            'description': 'Extract examples from official docs',
            'pros': ['High quality', 'Best practices', 'Well documented'],
            'cons': ['Limited volume', 'Mostly basic examples'],
            'effort': 'Low',
            'data_multiplier': '5-10x'
        },
        
        'code_augmentation': {
            'description': 'Modify existing code (rename variables, etc.)',
            'pros': ['Easy to implement', 'Preserves structure', 'Increases variety'],
            'cons': ['Limited new patterns', 'May introduce noise'],
            'effort': 'Very Low',
            'data_multiplier': '3-5x'
        }
    }
    
    return strategies

def should_add_more_data_or_train_more(current_epoch, loss_history):
    """Determine if you should add data or train more"""
    
    if len(loss_history) < 5:
        return "train_more", "Need more epochs to assess"
    
    # Check if loss is still decreasing
    recent_losses = loss_history[-5:]
    loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
    
    # Check for plateauing
    loss_variance = np.var(recent_losses)
    
    if loss_trend > -0.001 and loss_variance < 0.001:
        return "add_data", "Loss has plateaued - model needs more diverse data"
    elif loss_trend < -0.01:
        return "train_more", "Loss still decreasing - continue training"
    elif current_epoch < 20:
        return "train_more", "Early in training - give it more time"
    else:
        return "add_data", "Diminishing returns - time for more data"

if __name__ == "__main__":
    # Analyze training scenarios
    results = analyze_training_progress()
    
    print(f"\nAnalysis Results:")
    print(f"Small dataset final loss: {results['small_data_final_loss']:.3f}")
    print(f"Large dataset final loss: {results['large_data_final_loss']:.3f}")
    print(f"Improvement with more data: {results['improvement_ratio']:.2f}x better")
    
    # Show data expansion strategies
    print("\nData Expansion Strategies:")
    print("=" * 50)
    
    strategies = get_data_expansion_strategies()
    for name, strategy in strategies.items():
        print(f"\n{name.upper()}:")
        print(f"  Description: {strategy['description']}")
        print(f"  Effort: {strategy['effort']}")
        print(f"  Data increase: {strategy['data_multiplier']}")
        print(f"  Pros: {', '.join(strategy['pros'])}")
        print(f"  Cons: {', '.join(strategy['cons'])}")
    
    # Simulate current training decision
    print("\n" + "="*50)
    print("RECOMMENDATION FOR YOUR CURRENT MODEL:")
    print("="*50)
    
    # Simulate some loss history
    simulated_losses = [2.1, 1.8, 1.5, 1.2, 0.8, 0.6, 0.5, 0.48, 0.47, 0.46]
    decision, reason = should_add_more_data_or_train_more(10, simulated_losses)
    
    print(f"Current situation: {len(simulated_losses)} epochs, loss trend: {simulated_losses[-1]:.3f}")
    print(f"Decision: {decision.upper()}")
    print(f"Reason: {reason}")
    
    if decision == "add_data":
        print("\nRecommended next steps:")
        print("1. Try synthetic code generation (easiest)")
        print("2. Mine documentation examples")
        print("3. If still not enough, consider web scraping")
    else:
        print("\nRecommended next steps:")
        print("1. Continue training for 10-20 more epochs")
        print("2. Monitor for overfitting")
        print("3. Consider lowering learning rate")