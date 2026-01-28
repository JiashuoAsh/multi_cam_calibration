#!/usr/bin/env python3
"""
Unit tests for transformation utilities and SE(3) operations.
测试SE(3)变换相关函数的单元测试
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestSE3Transforms:
    """Test SE(3) transformation operations / 测试SE(3)变换操作"""
    
    def test_se3_exp_identity(self):
        """Test SE(3) exponential map at zero returns identity"""
        xi = np.zeros(6)
        T = self._se3_exp(xi)
        assert np.allclose(T, np.eye(4), atol=1e-10)
    
    def test_se3_exp_pure_translation(self):
        """Test SE(3) exponential map with pure translation"""
        xi = np.array([0, 0, 0, 1, 2, 3])  # [w, v] with w=0
        T = self._se3_exp(xi)
        
        # Rotation should be identity
        assert np.allclose(T[:3, :3], np.eye(3), atol=1e-10)
        # Translation should be v
        assert np.allclose(T[:3, 3], [1, 2, 3], atol=1e-10)
    
    def test_se3_exp_pure_rotation(self):
        """Test SE(3) exponential map with pure rotation"""
        # 90 degree rotation around Z axis
        theta = np.pi / 2
        xi = np.array([0, 0, theta, 0, 0, 0])
        T = self._se3_exp(xi)
        
        # Check rotation matrix properties
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)
        
        # Translation should be zero
        assert np.allclose(T[:3, 3], 0, atol=1e-10)
    
    def test_transform_composition(self):
        """Test transform composition: A_T_C = A_T_B @ B_T_C"""
        # Create test transforms
        A_T_B = self._create_transform([1, 0, 0], [0, 0, np.pi/2])
        B_T_C = self._create_transform([0, 1, 0], [0, 0, 0])
        
        # Compose transforms
        A_T_C = A_T_B @ B_T_C
        
        # Verify result
        # Point at (1, 0, 0) in C should be at (1, 1, 0) in A
        p_C = np.array([1, 0, 0, 1])
        p_A = A_T_C @ p_C
        expected = np.array([1, 1, 0, 1])
        assert np.allclose(p_A, expected, atol=1e-6)
    
    def test_transform_inverse(self):
        """Test transform inversion"""
        T = self._create_transform([1, 2, 3], [0.1, 0.2, 0.3])
        T_inv = np.linalg.inv(T)
        
        # T @ T_inv should be identity
        result = T @ T_inv
        assert np.allclose(result, np.eye(4), atol=1e-10)
        
        # Test on a point
        p = np.array([1, 2, 3, 1])
        p_transformed = T @ p
        p_recovered = T_inv @ p_transformed
        assert np.allclose(p, p_recovered, atol=1e-10)
    
    def test_rotation_matrix_properties(self):
        """Test rotation matrix properties (orthogonal, det=1)"""
        for _ in range(10):
            # Random rotation
            angles = np.random.uniform(-np.pi, np.pi, 3)
            T = self._create_transform([0, 0, 0], angles)
            R = T[:3, :3]
            
            # Orthogonality: R^T R = I
            assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)
            
            # Determinant = 1 (proper rotation)
            assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)
    
    # Helper methods
    def _se3_exp(self, xi: np.ndarray) -> np.ndarray:
        """SE(3) exponential map implementation for testing"""
        w = xi[:3]  # rotation part (axis-angle)
        v = xi[3:]  # translation part
        
        theta = np.linalg.norm(w)
        
        if theta < 1e-10:
            # Near identity
            R = np.eye(3)
            t = v
        else:
            # Rodrigues formula
            w_normalized = w / theta
            w_hat = self._skew_symmetric(w_normalized)
            
            R = (np.eye(3) + 
                 np.sin(theta) * w_hat + 
                 (1 - np.cos(theta)) * (w_hat @ w_hat))
            
            # Translation part (V matrix)
            V = (np.eye(3) + 
                 (1 - np.cos(theta)) / theta * w_hat +
                 (theta - np.sin(theta)) / theta * (w_hat @ w_hat))
            t = V @ v
        
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    
    def _skew_symmetric(self, w: np.ndarray) -> np.ndarray:
        """Create skew-symmetric matrix from vector"""
        return np.array([
            [0, -w[2], w[1]],
            [w[2], 0, -w[0]],
            [-w[1], w[0], 0]
        ])
    
    def _create_transform(self, translation, rotation_euler):
        """Create 4x4 transformation matrix from translation and euler angles"""
        from scipy.spatial.transform import Rotation
        
        R = Rotation.from_euler('XYZ', rotation_euler).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = translation
        return T


class TestIntrinsicParameters:
    """Test intrinsic camera parameter validity / 测试相机内参有效性"""
    
    def test_intrinsic_matrix_structure(self):
        """Test camera intrinsic matrix has correct structure"""
        K = np.array([
            [800, 0, 640],
            [0, 800, 480],
            [0, 0, 1]
        ])
        
        # Check structure
        assert K[2, 2] == 1  # Homogeneous coordinate
        assert K[1, 0] == 0  # Lower left should be zero
        assert K[2, 0] == 0 and K[2, 1] == 0  # Bottom row
        
        # Check positive focal lengths
        assert K[0, 0] > 0  # fx > 0
        assert K[1, 1] > 0  # fy > 0
    
    def test_intrinsic_parameter_ranges(self):
        """Test intrinsic parameters are in reasonable ranges"""
        # Typical camera parameters
        fx, fy = 800, 800
        cx, cy = 640, 480
        
        # Focal length should be positive and reasonable
        assert 100 < fx < 5000
        assert 100 < fy < 5000
        
        # Principal point should be within image
        assert 0 < cx < 2000
        assert 0 < cy < 2000
    
    def test_distortion_coefficient_validity(self):
        """Test distortion coefficients are reasonable"""
        dist = np.array([0.1, -0.05, 0.001, 0.0002, 0.01])
        
        # k1, k2 (radial distortion) should be small
        assert abs(dist[0]) < 1.0
        assert abs(dist[1]) < 1.0
        
        # p1, p2 (tangential distortion) should be very small
        assert abs(dist[2]) < 0.01
        assert abs(dist[3]) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
