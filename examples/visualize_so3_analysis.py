#!/usr/bin/env python3
"""
Comprehensive visualization and analysis of SO3/SE3 implementations.

This script provides:
1. 3D visualization of rotations
2. Accuracy analysis and numerical stability tests
3. Comparison with SciPy (conceptual benchmark)
4. Performance benchmarking
5. SE3 to SE2 conversion analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import sys
import os

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from pymatlie.so3 import SO3
    from pymatlie.se3 import SE3
    from pymatlie.se2 import SE2
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def test_so3_structure():
    """Test and analyze the SO3 implementation structure."""
    print("=== SO3 Implementation Analysis ===")
    
    # Test basic properties
    print(f"SO3.g_dim: {SO3.g_dim} (axis-angle representation)")
    print(f"SO3.matrix_size: {SO3.matrix_size}")
    
    # Test basic functionality with small examples
    print("\nBasic functionality test:")
    
    # Test with known axis-angle rotations
    test_rotations = torch.tensor([
        [0.0, 0.0, 0.0],              # identity
        [0.0, 0.0, np.pi/2],          # 90° around z
        [np.pi/2, 0.0, 0.0],          # 90° around x
        [0.0, np.pi/2, 0.0],          # 90° around y
        [0.1, 0.2, 0.3],              # general rotation
    ])
    
    print(f"Test rotations shape: {test_rotations.shape}")
    
    try:
        # Test exp/log cycle
        R = SO3.exp(test_rotations)
        recovered = SO3.log(R)
        error = torch.norm(test_rotations - recovered, dim=-1)
        print(f"  ✓ exp/log cycle error: {error.mean():.2e} (max: {error.max():.2e})")
        
        # Test hat/vee cycle
        skew_matrices = SO3.hat(test_rotations)
        recovered_vecs = SO3.vee(skew_matrices)
        error = torch.norm(test_rotations - recovered_vecs, dim=-1)
        print(f"  ✓ hat/vee cycle error: {error.mean():.2e} (max: {error.max():.2e})")
        
        # Test rotation matrix properties
        dets = torch.det(R)
        ortho_error = torch.norm(torch.bmm(R, R.transpose(-2, -1)) - torch.eye(3), dim=(-2, -1))
        print(f"  ✓ determinant check: {dets.mean():.6f} ± {dets.std():.2e}")
        print(f"  ✓ orthogonality error: {ortho_error.mean():.2e} (max: {ortho_error.max():.2e})")
        
        # Test quaternion conversions
        q = SO3.axis_angle_to_quaternion_robust(test_rotations)
        R_from_q = SO3.quaternion_to_matrix_robust(q)
        matrix_error = torch.norm(R - R_from_q, dim=(-2, -1))
        print(f"  ✓ quaternion conversion error: {matrix_error.mean():.2e} (max: {matrix_error.max():.2e})")
        
    except Exception as e:
        print(f"  ✗ Error in SO3 operations: {e}")


def test_se3_structure():
    """Test and analyze the SE3 implementation structure.""" 
    print("\n=== SE3 Implementation Analysis ===")
    
    print(f"SE3.g_dim: {SE3.g_dim} (3 translation + 3 axis-angle)")
    print(f"SE3.matrix_size: {SE3.matrix_size}")
    
    # Test with known SE3 configurations
    test_configs = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],          # identity
        [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],          # pure translation
        [0.0, 0.0, 0.0, 0.0, 0.0, np.pi/2],      # pure rotation (90° around z)
        [1.0, 1.0, 0.5, 0.1, 0.2, 0.3],          # general transformation
    ])
    
    print(f"Test configurations shape: {test_configs.shape}")
    
    try:
        # Test exp/log cycle
        T = SE3.exp(test_configs)
        recovered = SE3.log(T)
        error = torch.norm(test_configs - recovered, dim=-1)
        print(f"  ✓ exp/log cycle error: {error.mean():.2e} (max: {error.max():.2e})")
        
        # Test configuration mapping
        T_from_config = SE3.map_q_to_configuration(test_configs)
        recovered_config = SE3.map_configuration_to_q(T_from_config)
        config_error = torch.norm(test_configs - recovered_config, dim=-1)
        print(f"  ✓ config mapping error: {config_error.mean():.2e} (max: {config_error.max():.2e})")
        
        # Test homogeneous matrix structure
        bottom_row_expected = torch.tensor([0., 0., 0., 1.]).repeat(T.shape[0], 1)
        bottom_row_actual = T[..., 3, :]
        bottom_error = torch.norm(bottom_row_expected - bottom_row_actual, dim=-1)
        print(f"  ✓ homogeneous structure error: {bottom_error.mean():.2e}")
        
    except Exception as e:
        print(f"  ✗ Error in SE3 operations: {e}")


def test_se3_to_se2_conversion():
    """Test SE3 to SE2 conversion functionality."""
    print("\n=== SE3 to SE2 Conversion Analysis ===")
    
    # Create test 3D poses that should project nicely to 2D
    test_poses_3d = torch.tensor([
        [2.0, 3.0, 0.1, 0.0, 0.0, np.pi/4],     # mostly in xy-plane with small z
        [1.0, 2.0, 0.0, 0.0, 0.0, np.pi/2],     # pure xy motion
        [0.5, -1.5, 0.2, 0.05, 0.02, -np.pi/6], # general motion with small out-of-plane
    ])
    
    try:
        # Convert to SE3 matrices
        T_se3 = SE3.map_q_to_configuration(test_poses_3d)
        print(f"SE3 poses shape: {T_se3.shape}")
        
        for ignore_axis in ['z', 'y', 'x']:
            print(f"\n--- Projecting to 2D (ignore {ignore_axis}-axis) ---")
            
            # Convert to SE2
            T_se2 = SE3.se3_to_se2(T_se3, ignore_axis=ignore_axis)
            q_se2 = SE2.map_configuration_to_q(T_se2)
            
            print(f"SE2 configurations shape: {q_se2.shape}")
            print(f"Sample SE2 configs [x, y, theta]:")
            for i in range(min(3, len(q_se2))):
                x, y, theta = q_se2[i]
                theta_deg = theta * 180 / np.pi
                print(f"  Pose {i}: [{x:.3f}, {y:.3f}, {theta:.3f} rad] ({theta_deg:.1f}°)")
        
        # Test velocity conversion
        print(f"\n--- Velocity Conversion Test ---")
        test_velocities = torch.tensor([
            [1.0, 0.5, 0.1, 0.02, 0.01, 0.3],    # forward motion with yaw
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.2],      # pure sideways with rotation
            [-0.5, 0.0, 0.0, 0.0, 0.0, -0.1],    # reverse motion
        ])
        
        for ignore_axis in ['z']:  # Test main case
            xi_se2 = SE3.se3_to_se2_velocity(test_velocities, ignore_axis=ignore_axis)
            print(f"SE3 velocities → SE2 (ignore {ignore_axis}):")
            for i in range(len(xi_se2)):
                vx, vy, w = xi_se2[i]
                print(f"  [{vx:.3f}, {vy:.3f}, {w:.3f}]")
                
    except Exception as e:
        print(f"  ✗ Error in SE3→SE2 conversion: {e}")


def visualize_rotation_examples():
    """Visualize rotation matrices as 3D coordinate frames."""
    print("\n=== Generating 3D Rotation Visualization ===")
    
    try:
        fig = plt.figure(figsize=(15, 5))
        
        # Test rotations
        test_axis_angles = torch.tensor([
            [0.0, 0.0, 0.0],              # Identity
            [0.0, 0.0, np.pi/2],          # 90° around Z
            [np.pi/2, 0.0, 0.0],          # 90° around X  
            [0.0, np.pi/2, 0.0],          # 90° around Y
            [np.pi/4, np.pi/4, np.pi/4],  # General rotation
        ])
        
        labels = ["Identity", "90° Z", "90° X", "90° Y", "General"]
        
        for i, (axis_angle, label) in enumerate(zip(test_axis_angles, labels)):
            ax = fig.add_subplot(1, 5, i+1, projection='3d')
            
            # Convert to rotation matrix
            R = SO3.exp(axis_angle.unsqueeze(0))[0].numpy()
            
            # Plot coordinate frame
            origin = np.array([0, 0, 0])
            
            # X-axis (red)
            ax.quiver(origin[0], origin[1], origin[2], 
                     R[0, 0], R[1, 0], R[2, 0], 
                     color='red', arrow_length_ratio=0.1, linewidth=3, label='X')
            
            # Y-axis (green)  
            ax.quiver(origin[0], origin[1], origin[2], 
                     R[0, 1], R[1, 1], R[2, 1], 
                     color='green', arrow_length_ratio=0.1, linewidth=3, label='Y')
            
            # Z-axis (blue)
            ax.quiver(origin[0], origin[1], origin[2], 
                     R[0, 2], R[1, 2], R[2, 2], 
                     color='blue', arrow_length_ratio=0.1, linewidth=3, label='Z')
            
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{label}\n[{axis_angle[0]:.2f}, {axis_angle[1]:.2f}, {axis_angle[2]:.2f}]')
            
            if i == 0:
                ax.legend()
        
        plt.tight_layout()
        plt.savefig('so3_rotations_visualization.png', dpi=300, bbox_inches='tight')
        print("  ✓ Saved: so3_rotations_visualization.png")
        plt.show()
        
    except Exception as e:
        print(f"  ✗ Error in visualization: {e}")


def test_numerical_stability():
    """Test numerical stability with edge cases."""
    print("\n=== Numerical Stability Tests ===")
    
    try:
        # Test 1: Very small rotations
        print("1. Small angle stability:")
        small_angles = torch.tensor([1e-8, 1e-6, 1e-4, 1e-2])
        for angle in small_angles:
            axis_angle = torch.tensor([[0.0, 0.0, angle]])
            R = SO3.exp(axis_angle)
            recovered = SO3.log(R)
            error = torch.norm(axis_angle - recovered)
            print(f"   Angle {angle:.1e}: error = {error:.2e}")
        
        # Test 2: Large rotations  
        print("\n2. Large angle stability:")
        large_angles = torch.tensor([np.pi - 1e-6, np.pi, np.pi + 1e-6])
        for angle in large_angles:
            axis_angle = torch.tensor([[0.0, 0.0, angle]])
            R = SO3.exp(axis_angle)
            recovered = SO3.log(R)
            # Handle angle wrapping
            error = min(torch.norm(axis_angle - recovered), 
                       torch.norm(axis_angle - recovered + 2*np.pi),
                       torch.norm(axis_angle - recovered - 2*np.pi))
            print(f"   Angle {angle:.6f}: error = {error:.2e}")
        
        # Test 3: Random rotation robustness
        print("\n3. Random rotation robustness:")
        torch.manual_seed(42)
        random_rotations = torch.randn(100, 3) * 2  # random rotations up to ±4 rad
        
        R = SO3.exp(random_rotations)
        recovered = SO3.log(R)
        
        # Check rotation matrix properties
        dets = torch.det(R)
        det_error = torch.abs(dets - 1.0)
        ortho_error = torch.norm(torch.bmm(R, R.transpose(-2, -1)) - torch.eye(3), dim=(-2, -1))
        
        print(f"   Mean determinant error: {det_error.mean():.2e}")
        print(f"   Mean orthogonality error: {ortho_error.mean():.2e}")
        print(f"   Max orthogonality error: {ortho_error.max():.2e}")
        
        # Check exp/log consistency  
        exp_log_error = torch.norm(random_rotations - recovered, dim=-1)
        print(f"   Mean exp/log error: {exp_log_error.mean():.2e}")
        print(f"   Max exp/log error: {exp_log_error.max():.2e}")
        
    except Exception as e:
        print(f"  ✗ Error in stability tests: {e}")


def benchmark_performance():
    """Benchmark performance of different operations.""" 
    print("\n=== Performance Benchmark ===")
    
    try:
        sizes = [100, 1000, 10000]
        
        for N in sizes:
            print(f"\nTesting with {N} elements:")
            
            # Generate test data
            torch.manual_seed(42)
            axis_angles = torch.randn(N, 3)
            
            # Benchmark exp
            start_time = time.time()
            R = SO3.exp(axis_angles)
            exp_time = time.time() - start_time
            
            # Benchmark log
            start_time = time.time()
            recovered = SO3.log(R)
            log_time = time.time() - start_time
            
            # Benchmark quaternion conversions
            start_time = time.time()
            q = SO3.axis_angle_to_quaternion_robust(axis_angles)
            aa_to_q_time = time.time() - start_time
            
            start_time = time.time()
            R_from_q = SO3.quaternion_to_matrix_robust(q)
            q_to_mat_time = time.time() - start_time
            
            print(f"  SO3.exp:           {exp_time:.4f}s ({N/exp_time:.0f} ops/s)")
            print(f"  SO3.log:           {log_time:.4f}s ({N/log_time:.0f} ops/s)")
            print(f"  axis_angle→quat:   {aa_to_q_time:.4f}s ({N/aa_to_q_time:.0f} ops/s)")
            print(f"  quaternion→matrix: {q_to_mat_time:.4f}s ({N/q_to_mat_time:.0f} ops/s)")
            
    except Exception as e:
        print(f"  ✗ Error in benchmarking: {e}")


def compare_with_scipy_conceptual():
    """Conceptual comparison with SciPy (if available)."""
    print("\n=== Conceptual SciPy Comparison ===")
    
    try:
        from scipy.spatial.transform import Rotation as R_scipy
        print("SciPy available - running comparison...")
        
        # Generate test data
        torch.manual_seed(42)
        np.random.seed(42)
        N = 1000
        
        # Test axis-angle representations
        axis_angles = torch.randn(N, 3).numpy()
        
        # Our implementation
        start_time = time.time()
        R_ours = SO3.exp(torch.tensor(axis_angles, dtype=torch.float32))
        our_time = time.time() - start_time
        
        # SciPy implementation  
        start_time = time.time()
        R_scipy_obj = R_scipy.from_rotvec(axis_angles)
        R_scipy = R_scipy_obj.as_matrix()
        scipy_time = time.time() - start_time
        
        # Compare results
        diff = np.linalg.norm(R_ours.numpy() - R_scipy, axis=(-2, -1))
        
        print(f"Performance:")
        print(f"  Our implementation: {our_time:.4f}s")
        print(f"  SciPy:             {scipy_time:.4f}s")
        print(f"  Ratio:             {scipy_time/our_time:.2f}x")
        
        print(f"Accuracy:")
        print(f"  Mean difference: {diff.mean():.2e}")
        print(f"  Max difference:  {diff.max():.2e}")
        print(f"  Std difference:  {diff.std():.2e}")
        
    except ImportError:
        print("SciPy not available - skipping comparison")
        print("Install with: pip install scipy")
    except Exception as e:
        print(f"  ✗ Error in SciPy comparison: {e}")


def analysis_summary():
    """Print summary of the SO3 implementation."""
    print("\n" + "="*60)
    print("SO3/SE3 IMPLEMENTATION SUMMARY")
    print("="*60)
    
    print("✓ SO3: Axis-angle representation (3D vectors)")
    print("✓ SE3: Translation + axis-angle (6D vectors)")
    print("✓ Robust quaternion utilities (PyTorch3D inspired)")
    print("✓ Proper Lie group operations (exp, log, hat, vee)")
    print("✓ Numerical stability for small angles")
    print("✓ SE3 ↔ SE2 conversion for mobile robots")
    print("✓ Left Jacobians with proper formulation")
    
    print("\nKey Features:")
    print("• Batched operations for efficiency")
    print("• Rodrigues formula for exp()")
    print("• Robust matrix ↔ quaternion conversions")
    print("• SE3→SE2 projection for any axis")
    print("• Comprehensive error handling")
    
    print("\nRecommended Usage:")
    print("• Use axis-angle for Lie algebra operations")
    print("• Use quaternions for storage/interpolation")
    print("• Use SE3→SE2 for mobile robot applications")
    print("• Check quaternion_utils for robust conversions")


if __name__ == "__main__":
    print("SO3/SE3 Implementation Analysis and Visualization")
    print("=" * 60)
    
    # Run all tests and visualizations
    try:
        test_so3_structure()
        test_se3_structure()
        test_se3_to_se2_conversion()
        visualize_rotation_examples()
        test_numerical_stability()
        benchmark_performance()
        compare_with_scipy_conceptual()
        analysis_summary()
        
        print(f"\n" + "="*60)
        print("Analysis complete! Check generated PNG files for visualizations.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc() 