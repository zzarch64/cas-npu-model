#!/usr/bin/env python
"""
运行所有测试的脚本

使用方法:
    python test/run_all_tests.py              # 运行所有测试
    python test/run_all_tests.py --unit       # 只运行单元测试
    python test/run_all_tests.py --integration # 只运行集成测试
    python test/run_all_tests.py -v           # 详细输出
    python test/run_all_tests.py -q           # 安静模式
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# 添加扩展路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 获取测试目录
TEST_DIR = Path(__file__).parent
UNIT_DIR = TEST_DIR / "unit"
INTEGRATION_DIR = TEST_DIR / "integration"
TOOLS_DIR = TEST_DIR / "tools"


def run_test(test_file, verbose=False, quiet=False):
    """运行单个测试文件"""
    cmd = [sys.executable, str(test_file)]
    
    if verbose:
        cmd.append("-vv")
    elif quiet:
        cmd.append("-q")
    else:
        cmd.append("-v")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=quiet,
            text=True,
            cwd=TEST_DIR.parent
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def print_section(title, char="=", width=80):
    """打印章节标题"""
    print("\n" + char * width)
    print(title)
    print(char * width)


def print_result(name, passed, verbose=False):
    """打印测试结果"""
    status = "✓" if passed else "✗"
    print(f"  {status} {name}")
    if not passed and verbose:
        print(f"    (Failed)")


def main():
    parser = argparse.ArgumentParser(
        description="Run all CAS-NPU tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 运行所有测试
  python test/run_all_tests.py
  
  # 只运行单元测试
  python test/run_all_tests.py --unit
  
  # 只运行集成测试
  python test/run_all_tests.py --integration
  
  # 详细输出
  python test/run_all_tests.py -vv
  
  # 安静模式（只显示结果）
  python test/run_all_tests.py -q
        """
    )
    
    parser.add_argument(
        '--unit',
        action='store_true',
        help='Only run unit tests'
    )
    parser.add_argument(
        '--integration',
        action='store_true',
        help='Only run integration tests'
    )
    parser.add_argument(
        '--tools',
        action='store_true',
        help='Also run test tools (gradient_analyzer, etc.)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='count',
        default=0,
        help='Increase verbosity (-v: normal, -vv: verbose, -vvv: debug)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Quiet mode (only show results)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cas_npu:0',
        help='Device to use for tests (default: cas_npu:0)'
    )
    
    args = parser.parse_args()
    
    # 确定要运行的测试类型
    run_unit = not args.integration if args.unit else (not args.unit)
    run_integration = not args.unit if args.integration else (not args.integration)
    
    if not args.unit and not args.integration:
        # 默认运行所有
        run_unit = True
        run_integration = True
    
    verbose = args.verbose > 0
    quiet = args.quiet
    
    print_section("CAS-NPU Test Suite", "=")
    print(f"Running tests from: {TEST_DIR}")
    print(f"Unit tests: {'Yes' if run_unit else 'No'}")
    print(f"Integration tests: {'Yes' if run_integration else 'No'}")
    print(f"Tools: {'Yes' if args.tools else 'No'}")
    print(f"Verbosity: {'High' if verbose else ('Quiet' if quiet else 'Normal')}")
    
    results = {
        'unit': [],
        'integration': [],
        'tools': []
    }
    
    # 运行单元测试（递归查找子目录）
    if run_unit:
        print_section("Unit Tests", "-")
        unit_tests = []
        # 查找直接子目录中的测试文件
        for test_file in sorted(UNIT_DIR.glob("test_*.py")):
            unit_tests.append(test_file)
        # 查找子目录中的测试文件
        for subdir in UNIT_DIR.iterdir():
            if subdir.is_dir():
                for test_file in sorted(subdir.glob("test_*.py")):
                    unit_tests.append(test_file)
        
        if not unit_tests:
            print("  No unit tests found")
        else:
            for test_file in unit_tests:
                # 使用相对路径作为测试名称
                rel_path = test_file.relative_to(UNIT_DIR)
                test_name = str(rel_path).replace(os.sep, "/").replace(".py", "")
                if not quiet:
                    print(f"\nRunning {test_name}...")
                
                passed, stdout, stderr = run_test(test_file, verbose, quiet)
                results['unit'].append((test_name, passed))
                
                if not quiet:
                    print_result(test_name, passed, verbose)
                    if verbose and stdout:
                        print(stdout)
                elif not passed:
                    # 在安静模式下，只显示失败的测试
                    print_result(test_name, passed, verbose)
    
    # 运行集成测试（递归查找子目录）
    if run_integration:
        print_section("Integration Tests", "-")
        integration_tests = []
        # 查找直接子目录中的测试文件
        for test_file in sorted(INTEGRATION_DIR.glob("test_*.py")):
            integration_tests.append(test_file)
        # 查找子目录中的测试文件
        for subdir in INTEGRATION_DIR.iterdir():
            if subdir.is_dir():
                for test_file in sorted(subdir.glob("test_*.py")):
                    integration_tests.append(test_file)
        
        if not integration_tests:
            print("  No integration tests found")
        else:
            for test_file in integration_tests:
                # 使用相对路径作为测试名称
                rel_path = test_file.relative_to(INTEGRATION_DIR)
                test_name = str(rel_path).replace(os.sep, "/").replace(".py", "")
                if not quiet:
                    print(f"\nRunning {test_name}...")
                
                passed, stdout, stderr = run_test(test_file, verbose, quiet)
                results['integration'].append((test_name, passed))
                
                if not quiet:
                    print_result(test_name, passed, verbose)
                    if verbose and stdout:
                        print(stdout)
                elif not passed:
                    print_result(test_name, passed, verbose)
    
    # 运行工具（可选）
    if args.tools:
        print_section("Test Tools", "-")
        tools = sorted(TOOLS_DIR.glob("*.py"))
        
        # 排除 __init__.py 等
        tools = [t for t in tools if t.stem != "__init__" and not t.stem.startswith("_")]
        
        if not tools:
            print("  No test tools found")
        else:
            for tool_file in tools:
                tool_name = tool_file.stem
                if not quiet:
                    print(f"\nRunning {tool_name}...")
                
                passed, stdout, stderr = run_test(tool_file, verbose, quiet)
                results['tools'].append((tool_name, passed))
                
                if not quiet:
                    print_result(tool_name, passed, verbose)
                    if verbose and stdout:
                        print(stdout)
                elif not passed:
                    print_result(tool_name, passed, verbose)
    
    # 汇总结果
    print_section("Test Summary", "=")
    
    total_tests = 0
    total_passed = 0
    
    if results['unit']:
        print("\nUnit Tests:")
        for name, passed in results['unit']:
            total_tests += 1
            if passed:
                total_passed += 1
            print_result(name, passed, verbose)
    
    if results['integration']:
        print("\nIntegration Tests:")
        for name, passed in results['integration']:
            total_tests += 1
            if passed:
                total_passed += 1
            print_result(name, passed, verbose)
    
    if results['tools']:
        print("\nTest Tools:")
        for name, passed in results['tools']:
            total_tests += 1
            if passed:
                total_passed += 1
            print_result(name, passed, verbose)
    
    print("\n" + "=" * 80)
    print(f"Total: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("All tests passed! ✓")
        return 0
    else:
        print(f"{total_tests - total_passed} test(s) failed! ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
