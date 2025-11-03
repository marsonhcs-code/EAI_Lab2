"""
Model pruning implementation using PyTorch's pruning utilities
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN
import os
import copy


def get_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def count_parameters(model):
    """Count total and non-zero parameters"""
    total_params = 0
    nonzero_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Check if the module has been pruned (has a weight_orig attribute)
            if hasattr(module, 'weight_orig'):
                # For pruned modules, count non-zero in the effective weight
                mask = module.weight_mask if hasattr(module, 'weight_mask') else torch.ones_like(module.weight_orig)
                effective_weight = module.weight_orig * mask
                total_params += module.weight_orig.numel()
                nonzero_params += torch.count_nonzero(effective_weight).item()
            else:
                # For unpruned modules, count normally
                total_params += module.weight.numel()
                nonzero_params += torch.count_nonzero(module.weight).item()
            
            # Also count bias if present
            if module.bias is not None:
                total_params += module.bias.numel()
                nonzero_params += torch.count_nonzero(module.bias).item()
    
    return total_params, nonzero_params


def evaluate_model(model, device, test_loader):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def apply_magnitude_pruning(model, amount=0.3):
    """Apply global magnitude-based pruning"""
    parameters_to_prune = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    # Apply global unstructured pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    
    return model


def apply_structured_pruning(model, amount=0.3):
    """Apply structured pruning (prune entire filters/neurons)"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
        elif isinstance(module, nn.Linear):
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
    
    return model


def remove_pruning_reparameterization(model):
    """Remove pruning reparameterization to make pruning permanent"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            try:
                prune.remove(module, 'weight')
            except (ValueError, AttributeError):
                # Module was not pruned, skip
                pass
    return model


def main():
    """Main pruning demonstration"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Load pre-trained model
    print("Loading pre-trained model...")
    model = SimpleCNN().to(device)
    
    if os.path.exists('models/mnist_cnn.pth'):
        model.load_state_dict(torch.load('models/mnist_cnn.pth', map_location=device))
        print("Model loaded successfully")
    else:
        print("Warning: No pre-trained model found. Please run train.py first.")
        print("Continuing with randomly initialized model for demonstration...")
    
    # Evaluate original model
    print("\n" + "="*60)
    print("ORIGINAL MODEL")
    print("="*60)
    original_accuracy = evaluate_model(model, device, test_loader)
    original_size = get_model_size(model)
    total_params, nonzero_params = count_parameters(model)
    
    print(f"Accuracy: {original_accuracy:.2f}%")
    print(f"Model size: {original_size:.4f} MB")
    print(f"Total parameters: {total_params:,}")
    print(f"Non-zero parameters: {nonzero_params:,}")
    
    # Test different pruning amounts
    pruning_amounts = [0.3, 0.5, 0.7, 0.9]
    
    print("\n" + "="*60)
    print("MAGNITUDE-BASED PRUNING")
    print("="*60)
    
    for amount in pruning_amounts:
        # Create a copy of the model
        pruned_model = copy.deepcopy(model)
        
        # Apply magnitude pruning
        pruned_model = apply_magnitude_pruning(pruned_model, amount=amount)
        
        # Evaluate pruned model
        pruned_accuracy = evaluate_model(pruned_model, device, test_loader)
        total_params, nonzero_params = count_parameters(pruned_model)
        sparsity = 100. * (total_params - nonzero_params) / total_params
        
        print(f"\nPruning amount: {amount*100:.0f}%")
        print(f"Accuracy: {pruned_accuracy:.2f}% (Δ: {pruned_accuracy - original_accuracy:+.2f}%)")
        print(f"Sparsity: {sparsity:.2f}%")
        print(f"Non-zero parameters: {nonzero_params:,} / {total_params:,}")
    
    print("\n" + "="*60)
    print("STRUCTURED PRUNING")
    print("="*60)
    
    for amount in pruning_amounts[:3]:  # Test fewer amounts for structured pruning
        # Create a copy of the model
        pruned_model = copy.deepcopy(model)
        
        # Apply structured pruning
        pruned_model = apply_structured_pruning(pruned_model, amount=amount)
        
        # Evaluate pruned model
        pruned_accuracy = evaluate_model(pruned_model, device, test_loader)
        total_params, nonzero_params = count_parameters(pruned_model)
        sparsity = 100. * (total_params - nonzero_params) / total_params
        
        print(f"\nPruning amount: {amount*100:.0f}%")
        print(f"Accuracy: {pruned_accuracy:.2f}% (Δ: {pruned_accuracy - original_accuracy:+.2f}%)")
        print(f"Sparsity: {sparsity:.2f}%")
        print(f"Non-zero parameters: {nonzero_params:,} / {total_params:,}")
    
    # Save a pruned model (50% magnitude pruning)
    print("\n" + "="*60)
    print("SAVING PRUNED MODEL")
    print("="*60)
    final_pruned_model = copy.deepcopy(model)
    final_pruned_model = apply_magnitude_pruning(final_pruned_model, amount=0.5)
    final_pruned_model = remove_pruning_reparameterization(final_pruned_model)
    
    os.makedirs('models', exist_ok=True)
    torch.save(final_pruned_model.state_dict(), 'models/mnist_cnn_pruned_50.pth')
    print("Pruned model (50% magnitude pruning) saved to models/mnist_cnn_pruned_50.pth")
    
    final_accuracy = evaluate_model(final_pruned_model, device, test_loader)
    print(f"Final pruned model accuracy: {final_accuracy:.2f}%")


if __name__ == '__main__':
    main()
