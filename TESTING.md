# 测试指南 / Testing Guide

本文档说明如何运行和编写测试。

## 快速开始 / Quick Start

```bash
# 安装测试依赖 / Install test dependencies
pip install -r requirements.txt

# 运行所有测试 / Run all tests
pytest tests/ -v

# 运行特定测试 / Run specific tests
pytest tests/unit/test_transforms.py -v
pytest tests/integration/ -v

# 生成覆盖率报告 / Generate coverage report
pytest tests/ --cov=. --cov-report=html
```

## 测试结构 / Test Structure

```
tests/
├── unit/                    # 单元测试 / Unit tests
│   ├── __init__.py
│   └── test_transforms.py  # SE(3) 变换测试
├── integration/             # 集成测试 / Integration tests
│   ├── __init__.py
│   └── test_pipeline_basics.py  # 流水线基础测试
└── fixtures/                # 测试数据 / Test data
    └── (test images, configs)
```

## 当前测试覆盖 / Current Test Coverage

### 单元测试 / Unit Tests (9 tests)

✅ **SE(3) 变换测试** (`test_transforms.py`)
- `test_se3_exp_identity` - SE(3)指数映射恒等变换
- `test_se3_exp_pure_translation` - 纯平移变换
- `test_se3_exp_pure_rotation` - 纯旋转变换
- `test_transform_composition` - 变换组合
- `test_transform_inverse` - 变换求逆
- `test_rotation_matrix_properties` - 旋转矩阵性质

✅ **内参参数测试**
- `test_intrinsic_matrix_structure` - 内参矩阵结构
- `test_intrinsic_parameter_ranges` - 参数范围验证
- `test_distortion_coefficient_validity` - 畸变系数验证

### 集成测试 / Integration Tests (7 tests)

✅ **配置加载测试** (`test_pipeline_basics.py`)
- `test_config_file_structure` - 配置文件结构验证
- `test_board_parameters_valid` - 标定板参数验证

✅ **结果结构测试**
- `test_results_directory_exists` - 结果目录验证
- `test_intrinsics_result_structure` - 内参结果结构

✅ **变换链测试**
- `test_camera_to_base_transform_chain` - 相机到底盘变换链

✅ **错误处理测试**
- `test_invalid_transform_detection` - 无效变换检测
- `test_empty_image_list_handling` - 空图像列表处理

## 测试统计 / Test Statistics

```
总测试数 / Total Tests: 16
通过 / Passed: 16 (100%)
失败 / Failed: 0
跳过 / Skipped: 0
运行时间 / Runtime: ~0.4s
```

## 编写新测试 / Writing New Tests

### 单元测试模板 / Unit Test Template

```python
# tests/unit/test_my_module.py
import pytest
import numpy as np

class TestMyFunction:
    """测试 my_function / Test my_function"""
    
    def test_basic_case(self):
        """测试基本情况"""
        result = my_function(input_data)
        assert result == expected_output
    
    def test_edge_case(self):
        """测试边界情况"""
        with pytest.raises(ValueError):
            my_function(invalid_input)
```

### 集成测试模板 / Integration Test Template

```python
# tests/integration/test_my_pipeline.py
import pytest
from pathlib import Path

class TestMyPipeline:
    """测试 my_pipeline / Test my_pipeline"""
    
    def test_end_to_end(self):
        """测试端到端流程"""
        result = run_pipeline(config)
        assert result.success == True
        assert result.output_exists()
```

## 测试最佳实践 / Testing Best Practices

### ✅ 推荐 / Recommended

1. **每个函数至少一个测试** - 确保基本功能覆盖
2. **测试边界条件** - 空输入、极值、错误类型
3. **使用清晰的测试名称** - `test_transform_composition` 优于 `test_1`
4. **添加文档字符串** - 说明测试目的
5. **使用 fixtures** - 复用测试数据
6. **独立的测试** - 每个测试应该能独立运行

### ❌ 避免 / Avoid

1. **依赖外部文件** - 除非必要，使用内存数据
2. **测试间依赖** - 不要依赖其他测试的结果
3. **过度复杂** - 一个测试一个功能点
4. **忽略边界情况** - 空值、零、负数等

## 持续集成 / Continuous Integration

本项目配置了 GitHub Actions CI：

```yaml
# .github/workflows/ci.yml
- 自动运行所有测试
- 多操作系统测试 (Ubuntu, Windows, macOS)
- Python 3.8-3.11 支持
- 代码质量检查 (flake8, pylint)
```

## 测试覆盖率目标 / Coverage Goals

| 模块 / Module | 当前 / Current | 目标 / Target |
|---------------|----------------|---------------|
| transforms | 30% | 80% |
| detection | 10% | 70% |
| calibration | 5% | 75% |
| utils | 15% | 60% |
| **总体 / Overall** | **15%** | **70%+** |

## 添加测试覆盖 / Adding Test Coverage

优先级建议：

### 🔴 高优先级 / High Priority
1. SE(3) 变换函数（已完成 80%）
2. 配置加载与验证（已完成 60%）
3. 内参标定核心逻辑
4. 外参标定核心逻辑

### 🟡 中优先级 / Medium Priority
5. AprilTag 检测
6. 图像筛选
7. 结果保存与加载

### 🟢 低优先级 / Low Priority
8. 可视化功能
9. 辅助工具脚本
10. 文档生成

## 故障排查 / Troubleshooting

### 测试失败 / Tests Fail

```bash
# 详细输出 / Verbose output
pytest tests/ -vv

# 显示完整错误 / Show full errors
pytest tests/ --tb=long

# 在第一个失败时停止 / Stop at first failure
pytest tests/ -x
```

### 导入错误 / Import Errors

```bash
# 确保在项目根目录运行 / Run from project root
cd /path/to/multi_cam_calibration
pytest tests/

# 或添加项目路径 / Or add project path
PYTHONPATH=. pytest tests/
```

### 依赖问题 / Dependency Issues

```bash
# 重新安装依赖 / Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# 使用虚拟环境 / Use virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 资源 / Resources

- [pytest 文档](https://docs.pytest.org/)
- [pytest-cov 文档](https://pytest-cov.readthedocs.io/)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)

---

**最后更新 / Last Updated**: 2026-01-28  
**测试框架版本 / Test Framework Version**: pytest 7.0+
