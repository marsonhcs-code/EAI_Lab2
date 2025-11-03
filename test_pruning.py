"""
Test script to verify pruning functionality without requiring MNIST download
"""
import torch
import torch.nn as nn
from model import SimpleCNN
from prune import (
    get_model_size, 
    count_parameters, 
    apply_magnitude_pruning, 
    apply_structured_pruning,
    remove_pruning_reparameterization
)
import copy


def test_model_creation():
    """Test that model can be created"""
    print("Testing model creation...")
    model = SimpleCNN()
    print("✓ Model created successfully")
    
    # Test forward pass with dummy data
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"
    print("✓ Forward pass works correctly")
    return model


def test_model_metrics():
    """Test model size and parameter counting"""
    print("\nTesting model metrics...")
    model = SimpleCNN()
    
    size_mb = get_model_size(model)
    print(f"  Model size: {size_mb:.4f} MB")
    
    total_params, nonzero_params = count_parameters(model)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Non-zero parameters: {nonzero_params:,}")
    
    assert total_params == nonzero_params, "Unpruned model should have all non-zero parameters"
    print("✓ Model metrics calculated correctly")


def test_magnitude_pruning():
    """Test magnitude-based pruning"""
    print("\nTesting magnitude-based pruning...")
    model = SimpleCNN()
    
    original_total, original_nonzero = count_parameters(model)
    
    # Apply 50% pruning
    pruned_model = apply_magnitude_pruning(copy.deepcopy(model), amount=0.5)
    pruned_total, pruned_nonzero = count_parameters(pruned_model)
    
    sparsity = 100. * (pruned_total - pruned_nonzero) / pruned_total
    print(f"  Original parameters: {original_nonzero:,}")
    print(f"  Pruned parameters: {pruned_nonzero:,}")
    print(f"  Sparsity: {sparsity:.2f}%")
    
    assert pruned_nonzero < original_nonzero, "Pruning should reduce non-zero parameters"
    assert 45 <= sparsity <= 55, f"Expected ~50% sparsity, got {sparsity:.2f}%"
    print("✓ Magnitude pruning works correctly")


def test_structured_pruning():
    """Test structured pruning"""
    print("\nTesting structured pruning...")
    model = SimpleCNN()
    
    original_total, original_nonzero = count_parameters(model)
    
    # Apply 30% structured pruning
    pruned_model = apply_structured_pruning(copy.deepcopy(model), amount=0.3)
    pruned_total, pruned_nonzero = count_parameters(pruned_model)
    
    sparsity = 100. * (pruned_total - pruned_nonzero) / pruned_total
    print(f"  Original parameters: {original_nonzero:,}")
    print(f"  Pruned parameters: {pruned_nonzero:,}")
    print(f"  Sparsity: {sparsity:.2f}%")
    
    assert pruned_nonzero < original_nonzero, "Pruning should reduce non-zero parameters"
    print("✓ Structured pruning works correctly")


def test_pruning_removal():
    """Test removing pruning reparameterization"""
    print("\nTesting pruning reparameterization removal...")
    model = SimpleCNN()
    
    # Apply pruning
    pruned_model = apply_magnitude_pruning(copy.deepcopy(model), amount=0.5)
    
    # Check that pruning masks exist
    has_mask_before = any('weight_mask' in name for name, _ in pruned_model.named_buffers())
    print(f"  Has pruning masks before removal: {has_mask_before}")
    assert has_mask_before, "Pruned model should have weight masks"
    
    # Remove pruning
    final_model = remove_pruning_reparameterization(pruned_model)
    
    # Check that masks are removed
    has_mask_after = any('weight_mask' in name for name, _ in final_model.named_buffers())
    print(f"  Has pruning masks after removal: {has_mask_after}")
    assert not has_mask_after, "Pruning masks should be removed"
    
    print("✓ Pruning reparameterization removal works correctly")


def test_forward_pass_after_pruning():
    """Test that model can still do forward pass after pruning"""
    print("\nTesting forward pass after pruning...")
    model = SimpleCNN()
    dummy_input = torch.randn(4, 1, 28, 28)
    
    # Test with magnitude pruning
    pruned_model = apply_magnitude_pruning(copy.deepcopy(model), amount=0.7)
    output = pruned_model(dummy_input)
    assert output.shape == (4, 10), f"Expected output shape (4, 10), got {output.shape}"
    print("  ✓ Magnitude pruned model forward pass works")
    
    # Test with structured pruning
    pruned_model = apply_structured_pruning(copy.deepcopy(model), amount=0.3)
    output = pruned_model(dummy_input)
    assert output.shape == (4, 10), f"Expected output shape (4, 10), got {output.shape}"
    print("  ✓ Structured pruned model forward pass works")
    
    print("✓ All forward passes work correctly after pruning")


def main():
    """Run all tests"""
    print("="*60)
    print("RUNNING PRUNING TESTS")
    print("="*60)
    
    try:
        test_model_creation()
        test_model_metrics()
        test_magnitude_pruning()
        test_structured_pruning()
        test_pruning_removal()
        test_forward_pass_after_pruning()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
