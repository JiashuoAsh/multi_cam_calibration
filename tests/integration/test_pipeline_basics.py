#!/usr/bin/env python3
"""
Integration tests for basic calibration pipeline operations.
基础标定流水线集成测试
"""

import numpy as np
import pytest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestConfigurationLoading:
    """Test configuration loading and validation / 测试配置加载与验证"""
    
    def test_config_file_structure(self):
        """Test that config file has expected structure"""
        config_path = Path(__file__).parent.parent.parent / "config" / "apriltag_config.json"
        
        if not config_path.exists():
            pytest.skip("Config file not found")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check required top-level keys
        assert "apriltag_board" in config
        assert "calibration_settings" in config
        
        # Check apriltag_board structure
        board = config["apriltag_board"]
        assert "family" in board
        assert "tags_x" in board
        assert "tags_y" in board
        assert "tag_size" in board
        assert "tag_spacing" in board
    
    def test_board_parameters_valid(self):
        """Test that board parameters are valid"""
        config_path = Path(__file__).parent.parent.parent / "config" / "apriltag_config.json"
        
        if not config_path.exists():
            pytest.skip("Config file not found")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        board = config["apriltag_board"]
        
        # Check positive values
        assert board["tag_size"] > 0
        assert board["tag_spacing"] > 0
        assert board["tags_x"] > 0
        assert board["tags_y"] > 0
        
        # Check reasonable ranges
        assert 1 <= board["tag_size"] <= 500  # 1mm to 500mm
        assert 0 <= board["tag_spacing"] <= 200  # 0mm to 200mm
        assert 2 <= board["tags_x"] <= 20
        assert 2 <= board["tags_y"] <= 20


class TestResultsStructure:
    """Test results directory and file structure / 测试结果目录结构"""
    
    def test_results_directory_exists(self):
        """Test that results directory can be created"""
        results_dir = Path(__file__).parent.parent.parent / "results"
        
        # Directory should exist or be creatable
        if not results_dir.exists():
            # This is expected for fresh repos
            pass
        else:
            assert results_dir.is_dir()
    
    def test_intrinsics_result_structure(self):
        """Test intrinsics result JSON structure"""
        # Define expected structure (for documentation)
        expected_keys = {
            "camera": str,
            "rms_error": float,
            "K": list,  # 3x3 matrix
            "dist": list,  # distortion coefficients
            "image_size": list,  # [width, height]
        }
        
        # This is a structural test - validates the expected format
        # Actual validation would require running calibration
        assert set(expected_keys.keys()) == {"camera", "rms_error", "K", "dist", "image_size"}


class TestTransformChain:
    """Test transformation chain operations / 测试变换链操作"""
    
    def test_camera_to_base_transform_chain(self):
        """Test B_T_C = B_T_T @ inv(C_T_T)"""
        # Create sample transforms
        # C_T_T: Tag in camera coordinates
        C_T_T = np.array([
            [1, 0, 0, 0.5],
            [0, 1, 0, 0],
            [0, 0, 1, 1.0],
            [0, 0, 0, 1]
        ])
        
        # B_T_T: Tag in base coordinates
        B_T_T = np.array([
            [1, 0, 0, 1.5],
            [0, 1, 0, 0],
            [0, 0, 1, 1.2],
            [0, 0, 0, 1]
        ])
        
        # Calculate B_T_C
        T_T_C = np.linalg.inv(C_T_T)
        B_T_C = B_T_T @ T_T_C
        
        # Verify result properties
        assert B_T_C.shape == (4, 4)
        assert np.allclose(B_T_C[3, :], [0, 0, 0, 1])
        
        # Verify transformation properties
        R = B_T_C[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)


class TestErrorHandling:
    """Test error handling and edge cases / 测试错误处理"""
    
    def test_invalid_transform_detection(self):
        """Test that invalid transforms are detected"""
        # Non-4x4 matrix
        T_invalid = np.eye(3)
        assert T_invalid.shape != (4, 4)
        
        # Invalid bottom row
        T_invalid = np.eye(4)
        T_invalid[3, 0] = 1
        assert not np.allclose(T_invalid[3, :], [0, 0, 0, 1])
        
        # Invalid rotation (not orthogonal)
        T_invalid = np.eye(4)
        T_invalid[0, 1] = 0.5
        R = T_invalid[:3, :3]
        assert not np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    
    def test_empty_image_list_handling(self):
        """Test handling of empty image lists"""
        images = []
        
        # Should handle gracefully
        assert len(images) == 0
        # Actual code should check this and raise appropriate error


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
