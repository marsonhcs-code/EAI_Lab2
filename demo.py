"""
Complete demonstration of model pruning workflow
This script shows the entire process without requiring MNIST download
"""
import torch
import torch.nn as nn
from model import SimpleCNN
from prune import (
    apply_magnitude_pruning,
    apply_structured_pruning,
    count_parameters,
    get_model_size,
    remove_pruning_reparameterization
)
import copy


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(title)
    print("="*80)


def demonstrate_basic_pruning():
    """Demonstrate basic pruning workflow"""
    print_section("BASIC PRUNING DEMONSTRATION")
    
    # Create model
    print("\n1. Creating base model...")
    model = SimpleCNN()
    total_params, nonzero_params = count_parameters(model)
    size_mb = get_model_size(model)
    
    print(f"   ✓ Model created")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Model size: {size_mb:.4f} MB")
    
    # Apply pruning
    print("\n2. Applying 50% magnitude-based pruning...")
    pruned_model = apply_magnitude_pruning(copy.deepcopy(model), amount=0.5)
    
    pruned_total, pruned_nonzero = count_parameters(pruned_model)
    sparsity = 100.0 * (pruned_total - pruned_nonzero) / pruned_total
    
    print(f"   ✓ Pruning applied")
    print(f"   - Non-zero parameters: {pruned_nonzero:,} / {pruned_total:,}")
    print(f"   - Sparsity: {sparsity:.2f}%")
    print(f"   - Compression ratio: {pruned_total / pruned_nonzero:.2f}x")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    dummy_input = torch.randn(1, 1, 28, 28)
    output = pruned_model(dummy_input)
    
    print(f"   ✓ Forward pass successful")
    print(f"   - Input shape: {dummy_input.shape}")
    print(f"   - Output shape: {output.shape}")
    print(f"   - Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Make pruning permanent
    print("\n4. Making pruning permanent...")
    final_model = remove_pruning_reparameterization(pruned_model)
    
    has_mask = any('weight_mask' in name for name, _ in final_model.named_buffers())
    print(f"   ✓ Pruning masks removed")
    print(f"   - Model ready for deployment: {not has_mask}")


def demonstrate_pruning_levels():
    """Demonstrate different pruning levels"""
    print_section("COMPARING DIFFERENT PRUNING LEVELS")
    
    model = SimpleCNN()
    original_total, _ = count_parameters(model)
    
    pruning_levels = [0.2, 0.4, 0.6, 0.8, 0.95]
    
    print(f"\n{'Pruning':<10} {'Non-zero Params':<20} {'Sparsity':<12} {'Compression'}")
    print("-" * 80)
    
    for level in pruning_levels:
        pruned = apply_magnitude_pruning(copy.deepcopy(model), amount=level)
        total, nonzero = count_parameters(pruned)
        sparsity = 100.0 * (total - nonzero) / total
        compression = total / nonzero
        
        print(f"{level*100:>6.0f}%    {nonzero:>12,}       {sparsity:>6.2f}%      {compression:>5.2f}x")


def demonstrate_pruning_methods():
    """Compare magnitude vs structured pruning"""
    print_section("COMPARING PRUNING METHODS")
    
    model = SimpleCNN()
    
    print("\nApplying 40% pruning with different methods:\n")
    
    # Magnitude pruning
    print("1. Magnitude-based Pruning:")
    mag_pruned = apply_magnitude_pruning(copy.deepcopy(model), amount=0.4)
    mag_total, mag_nonzero = count_parameters(mag_pruned)
    mag_sparsity = 100.0 * (mag_total - mag_nonzero) / mag_total
    
    print(f"   - Sparsity achieved: {mag_sparsity:.2f}%")
    print(f"   - Non-zero params: {mag_nonzero:,}")
    print(f"   - Prunes individual weights globally")
    print(f"   - Best for: Maximum sparsity with minimal accuracy loss")
    
    # Structured pruning
    print("\n2. Structured Pruning:")
    struct_pruned = apply_structured_pruning(copy.deepcopy(model), amount=0.4)
    struct_total, struct_nonzero = count_parameters(struct_pruned)
    struct_sparsity = 100.0 * (struct_total - struct_nonzero) / struct_total
    
    print(f"   - Sparsity achieved: {struct_sparsity:.2f}%")
    print(f"   - Non-zero params: {struct_nonzero:,}")
    print(f"   - Prunes entire filters/neurons")
    print(f"   - Best for: Direct speedup without specialized hardware")


def demonstrate_layer_analysis():
    """Show per-layer pruning statistics"""
    print_section("LAYER-BY-LAYER ANALYSIS")
    
    model = SimpleCNN()
    pruned_model = apply_magnitude_pruning(model, amount=0.5)
    
    print("\nPer-layer statistics with 50% global pruning:\n")
    print(f"{'Layer':<15} {'Type':<10} {'Total':<12} {'Non-zero':<12} {'Sparsity'}")
    print("-" * 80)
    
    for name, module in pruned_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            layer_type = "Conv2d" if isinstance(module, nn.Conv2d) else "Linear"
            
            if hasattr(module, 'weight_orig'):
                mask = module.weight_mask if hasattr(module, 'weight_mask') else torch.ones_like(module.weight_orig)
                effective_weight = module.weight_orig * mask
                total = module.weight_orig.numel()
                nonzero = torch.count_nonzero(effective_weight).item()
            else:
                total = module.weight.numel()
                nonzero = torch.count_nonzero(module.weight).item()
            
            sparsity = 100.0 * (total - nonzero) / total
            print(f"{name:<15} {layer_type:<10} {total:>10,}  {nonzero:>10,}  {sparsity:>6.2f}%")
    
    print("\nNote: Different layers achieve different sparsity levels due to")
    print("      varying parameter counts and importance for the task.")


def main():
    """Run all demonstrations"""
    print("\n" + "="*80)
    print(" "*20 + "MODEL PRUNING COMPLETE DEMONSTRATION")
    print("="*80)
    print("\nThis script demonstrates neural network pruning techniques")
    print("without requiring actual training or dataset downloads.")
    
    demonstrate_basic_pruning()
    demonstrate_pruning_levels()
    demonstrate_pruning_methods()
    demonstrate_layer_analysis()
    
    print_section("SUMMARY")
    print("""
Key Takeaways:

1. Model pruning reduces parameter count while maintaining functionality
2. Different pruning levels offer trade-offs between size and accuracy
3. Magnitude-based pruning: High sparsity, needs sparse ops for speedup
4. Structured pruning: Direct speedup, works on standard hardware
5. Layers with more parameters can typically handle more pruning

Next Steps:

- Run 'python test_pruning.py' to verify implementation
- Run 'python visualize.py' for detailed impact analysis
- Run 'python train.py' to train on actual MNIST data
- Run 'python prune.py' to evaluate pruned model accuracy

For production deployment, consider:
- Iterative pruning with fine-tuning
- Quantization combined with pruning
- Hardware-specific optimization
""")
    
    print("="*80)


if __name__ == '__main__':
    main()
