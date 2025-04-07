# Assignement 1

# 1. Setup Environment
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, PReLU

# 2. Data Preparation
def prepare_data():
    """Generate and standardize synthetic data"""
    print("\nGenerating synthetic data...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        random_state=42
    )
    X = (X - X.mean())/X.std()  # Standardize data
    print("Data shape:", X.shape)
    print("First sample (normalized):", X[0][:5])
    return X, y

# 3. Activation Functions Comparison
def compare_activations(X, y):
    """Compare sigmoid, tanh, and ReLU performance"""
    print("\nTraining models with different activations...")
    
    def train_model(activation):
        """Train simple 2-hidden-layer DNN"""
        model = Sequential([
            Dense(64, input_shape=(20,), activation=activation),
            Dense(32, activation=activation),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        history = model.fit(X, y, epochs=50, verbose=0)
        return history.history['loss']
    
    # Train and compare
    activations = ['sigmoid', 'tanh', 'relu']
    results = {a: train_model(a) for a in activations}
    
    # Plot results
    plt.figure(figsize=(10,6))
    for act, loss in results.items():
        plt.plot(loss, label=f'{act}')
    plt.title('Training Loss: Sigmoid vs Tanh vs ReLU')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('activation_comparison.png')
    plt.close()
    print("Saved activation comparison plot: activation_comparison.png")

# 4. ReLU Variants Comparison
def compare_relu_variants(X, y):
    """Compare ReLU, LeakyReLU, and PReLU"""
    print("\nTraining ReLU variants...")
    
    def train_variant(variant):
        """Train model with specific ReLU variant"""
        model = Sequential()
        model.add(Dense(64, input_shape=(20,)))
        
        if variant == 'leaky':
            model.add(LeakyReLU(alpha=0.1))
        elif variant == 'prelu':
            model.add(PReLU())
        else:
            model.add(Dense(64, activation='relu'))
        
        model.add(Dense(32))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model.fit(X, y, epochs=50, verbose=0).history['loss']
    
    # Train and compare
    variants = ['relu', 'leaky', 'prelu']
    results = {v: train_variant(v) for v in variants}
    
    # Plot results
    plt.figure(figsize=(10,6))
    for v, loss in results.items():
        plt.plot(loss, label=f'{v}')
    plt.title('ReLU Variants Performance')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('relu_variants.png')
    plt.close()
    print("Saved ReLU variants plot: relu_variants.png")

# 5. Network Depth Analysis
def analyze_depth(X, y):
    """Test impact of different network depths"""
    print("\nTesting different network depths...")
    
    depths = [1, 2, 3, 4, 5]
    results = {}
    
    for depth in depths:
        model = Sequential()
        for _ in range(depth):
            model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy')
        results[depth] = model.fit(X, y, epochs=50, verbose=0).history['loss']
    
    # Plot results
    plt.figure(figsize=(10,6))
    for depth, loss in results.items():
        plt.plot(loss, label=f'Depth {depth}')
    plt.title('Network Depth Impact')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('depth_analysis.png')
    plt.close()
    print("Saved depth analysis plot: depth_analysis.png")

# 6. Main Execution
if __name__ == "__main__":
    # Step 1: Prepare data
    X, y = prepare_data()
    
    # Step 2: Activation comparison
    compare_activations(X, y)
    
    # Step 3: ReLU variants
    compare_relu_variants(X, y)
    
    # Step 4: Depth analysis
    analyze_depth(X, y)
    
    print("\nExperiment complete! Check generated plots:")
    print("- activation_comparison.png")
    print("- relu_variants.png")
    print("- depth_analysis.png")
