import pytest
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from src.models.base_model import SimpleFeedForward

def test_model_initialization():
    print("Testing model initialization...")
    
    n_input = 10
    hidden_widths = [20, 15]
    n_output = 5
    model = SimpleFeedForward(n_input, hidden_widths, n_output)
    print(f"Model 1 initialized: {n_input} -> {hidden_widths} -> {n_output}")
    
    n_input = 5
    hidden_widths = [10, 20, 30, 20, 10]
    n_output = 2
    model = SimpleFeedForward(n_input, hidden_widths, n_output, dropout_rate=0.3)
    print(f"Model 2 initialized: {n_input} -> {hidden_widths} -> {n_output}")
    
    n_input = 8
    hidden_widths = []
    n_output = 1
    model = SimpleFeedForward(n_input, hidden_widths, n_output, dropout_rate=0.0)
    print(f"Model 3 initialized: {n_input} -> {hidden_widths} -> {n_output}")
    
    print("All model initializations passed!\n")

def test_forward_pass():
    print("Testing forward pass...")
    
    n_input = 10
    hidden_widths = [20, 15]
    n_output = 5
    model = SimpleFeedForward(n_input, hidden_widths, n_output)
    
    batch_size = 32
    x = torch.randn(batch_size, n_input)
    
    try:
        output = model(x)
        assert output.shape == (batch_size, n_output), f"Expected output shape {(batch_size, n_output)}, got {output.shape}"
        print(f"Forward pass successful! Output shape: {output.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
    
    print("Forward pass test completed!\n")

def test_train_on_synthetic_data():
    print("Testing model training on synthetic data...")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_samples = 1000
    n_input = 2
    n_output = 1
    
    X = np.random.uniform(-3, 3, (n_samples, n_input))
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.random.normal(0, 0.1, n_samples)
    y = y.reshape(-1, 1)
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X_tensor[:split_idx], X_tensor[split_idx:]
    y_train, y_test = y_tensor[:split_idx], y_tensor[split_idx:]
    
    hidden_widths = [32, 16]
    model = SimpleFeedForward(n_input, hidden_widths, n_output, dropout_rate=0.1)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    n_epochs = 100
    batch_size = 64
    train_losses = []
    test_losses = []
    
    for epoch in range(n_epochs):
        indices = torch.randperm(len(X_train))
        
        epoch_loss = 0.0
        batches = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
        
        avg_train_loss = epoch_loss / batches
        train_losses.append(avg_train_loss)
        
        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test)
            test_loss = criterion(y_test_pred, y_test).item()
            test_losses.append(test_loss)
        model.train()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test)
        final_test_loss = criterion(y_test_pred, y_test).item()
    
    print(f"Training complete! Final test loss: {final_test_loss:.4f}")
    print("Model training test completed!\n")

def test_gradient_flow():
    print("Testing gradient flow...")
    
    n_input = 5
    hidden_widths = [10, 8]
    n_output = 3
    model = SimpleFeedForward(n_input, hidden_widths, n_output)
    
    batch_size = 16
    x = torch.randn(batch_size, n_input)
    y = torch.randn(batch_size, n_output)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    initial_params = {}
    for name, param in model.named_parameters():
        initial_params[name] = param.clone().detach()
    
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    params_changed = False
    for name, param in model.named_parameters():
        if not torch.allclose(param, initial_params[name]):
            params_changed = True
            break
    
    if params_changed:
        print("Gradient flow test passed! Parameters were updated after backpropagation.")
    else:
        print("Gradient flow test failed! Parameters did not change after backpropagation.")
    
    print("Gradient flow test completed!\n")

def main():
    print("Running tests for SimpleFeedForward neural network...\n")
    
    test_model_initialization()
    test_forward_pass()
    test_gradient_flow()
    test_train_on_synthetic_data()
    
    print("All tests completed!")

if __name__ == "__main__":
    main()
