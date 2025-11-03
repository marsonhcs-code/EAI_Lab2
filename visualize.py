"""
Visualization script for model pruning results
"""
import torch
import torch.nn as nn
from model import SimpleCNN
from prune import apply_magnitude_pruning, count_parameters
import copy


def visualize_pruning_impact():
    """Create a simple text-based visualization of pruning impact"""
    model = SimpleCNN()
    
    print("="*80)
    print("MODEL PRUNING IMPACT ANALYSIS")
    print("="*80)
    print()
    
    # Original model
    total_params, nonzero_params = count_parameters(model)
    
    print("ORIGINAL MODEL:")
    print("-" * 80)
    print(f"  Total Parameters:     {total_params:,}")
    print(f"  Non-zero Parameters:  {nonzero_params:,}")
    print(f"  Sparsity:             0.00%")
    print()
    
    # Different pruning levels
    pruning_levels = [0.3, 0.5, 0.7, 0.9]
    
    print("MAGNITUDE PRUNING RESULTS:")
    print("-" * 80)
    print(f"{'Pruning %':<12} {'Non-zero Params':<20} {'Sparsity':<12} {'Compression'}")
    print("-" * 80)
    
    for amount in pruning_levels:
        pruned_model = copy.deepcopy(model)
        pruned_model = apply_magnitude_pruning(pruned_model, amount=amount)
        
        total, nonzero = count_parameters(pruned_model)
        sparsity = 100.0 * (total - nonzero) / total
        compression = total / nonzero
        
        # Create visual bar
        bar_length = 40
        filled = int((nonzero / total) * bar_length)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"{amount*100:>6.0f}%      {nonzero:>12,}       {sparsity:>6.2f}%      {compression:>5.2f}x")
    
    print()
    print("="*80)
    print()
    
    # Layer-wise analysis for 50% pruning
    print("LAYER-WISE ANALYSIS (50% Pruning):")
    print("-" * 80)
    pruned_model = apply_magnitude_pruning(copy.deepcopy(model), amount=0.5)
    
    print(f"{'Layer':<20} {'Total Params':<15} {'Non-zero':<15} {'Sparsity'}")
    print("-" * 80)
    
    for name, module in pruned_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight_orig'):
                mask = module.weight_mask if hasattr(module, 'weight_mask') else torch.ones_like(module.weight_orig)
                effective_weight = module.weight_orig * mask
                total = module.weight_orig.numel()
                nonzero = torch.count_nonzero(effective_weight).item()
                sparsity = 100.0 * (total - nonzero) / total
            else:
                total = module.weight.numel()
                nonzero = torch.count_nonzero(module.weight).item()
                sparsity = 0.0
            
            print(f"{name:<20} {total:>12,}   {nonzero:>12,}   {sparsity:>6.2f}%")
    
    print()
    print("="*80)


def compare_pruning_methods():
    """Compare magnitude vs structured pruning"""
    model = SimpleCNN()
    
    print()
    print("PRUNING METHOD COMPARISON (30% Target):")
    print("="*80)
    
    from prune import apply_structured_pruning
    
    # Magnitude pruning
    mag_pruned = apply_magnitude_pruning(copy.deepcopy(model), amount=0.3)
    mag_total, mag_nonzero = count_parameters(mag_pruned)
    mag_sparsity = 100.0 * (mag_total - mag_nonzero) / mag_total
    
    # Structured pruning
    struct_pruned = apply_structured_pruning(copy.deepcopy(model), amount=0.3)
    struct_total, struct_nonzero = count_parameters(struct_pruned)
    struct_sparsity = 100.0 * (struct_total - struct_nonzero) / struct_total
    
    print()
    print("Magnitude-based Pruning:")
    print("-" * 80)
    print(f"  Non-zero parameters: {mag_nonzero:,} / {mag_total:,}")
    print(f"  Actual sparsity: {mag_sparsity:.2f}%")
    print(f"  Advantages: High sparsity, simple implementation")
    print(f"  Requires: Sparse matrix support for acceleration")
    
    print()
    print("Structured Pruning:")
    print("-" * 80)
    print(f"  Non-zero parameters: {struct_nonzero:,} / {struct_total:,}")
    print(f"  Actual sparsity: {struct_sparsity:.2f}%")
    print(f"  Advantages: Direct speedup, no special hardware")
    print(f"  May require: More careful tuning")
    
    print()
    print("="*80)


def main():
    """Run visualizations"""
    visualize_pruning_impact()
    compare_pruning_methods()
    
    print()
    print("NOTE: For actual accuracy metrics, train a model first using:")
    print("  python train.py")
    print("Then run:")
    print("  python prune.py")
    print()


if __name__ == '__main__':
    main()
