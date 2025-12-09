#!/usr/bin/env python3
"""
GPU Implementation Comparison Script
Compares GPU implementations (OpenACC, CUDA Python, CUDA C++) with CPU ground truth
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

def print_section(text: str):
    """Print formatted section"""
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{'-'*len(text)}{Colors.ENDC}")

def compare_images(img1: np.ndarray, img2: np.ndarray) -> Tuple[bool, float, float]:
    """
    Compare two images pixel by pixel
    Returns: (are_identical, max_diff, mean_diff)
    """
    if img1 is None or img2 is None:
        return False, float('inf'), float('inf')
    
    if img1.shape != img2.shape:
        return False, float('inf'), float('inf')
    
    # Calculate differences
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Images are identical if all pixels are exactly the same
    are_identical = (max_diff <= 7.5)  # Allow small tolerance for minor differences
    
    return are_identical, max_diff, mean_diff

def check_implementation(impl_name: str, impl_dir: Path, ground_truth_dir: Path, 
                        expected_outputs: List[str], output_mapping: Dict[str, str] = None) -> Dict:
    """
    Check a single GPU implementation against ground truth
    output_mapping: Optional dict mapping GPU output filename -> CPU ground truth filename
    Returns dictionary with results
    """
    results = {
        'implementation': impl_name,
        'total': len(expected_outputs),
        'passed': 0,
        'failed': 0,
        'missing': 0,
        'details': []
    }
    
    for output_file in expected_outputs:
        gpu_path = impl_dir / output_file
        # Use mapping if provided, otherwise assume same filename
        ground_truth_file = output_mapping.get(output_file, output_file) if output_mapping else output_file
        cpu_path = ground_truth_dir / ground_truth_file
        
        # Check if files exist
        if not cpu_path.exists():
            results['details'].append({
                'file': output_file,
                'status': 'cpu_missing',
                'message': 'Ground truth file not found'
            })
            results['missing'] += 1
            continue
        
        if not gpu_path.exists():
            results['details'].append({
                'file': output_file,
                'status': 'gpu_missing',
                'message': 'GPU output file not found'
            })
            results['missing'] += 1
            continue
        
        # Load and compare images
        cpu_img = cv2.imread(str(cpu_path), cv2.IMREAD_GRAYSCALE)
        gpu_img = cv2.imread(str(gpu_path), cv2.IMREAD_GRAYSCALE)
        
        are_identical, max_diff, mean_diff = compare_images(cpu_img, gpu_img)
        
        if are_identical:
            results['details'].append({
                'file': output_file,
                'status': 'passed',
                'max_diff': max_diff,
                'mean_diff': mean_diff
            })
            results['passed'] += 1
        else:
            results['details'].append({
                'file': output_file,
                'status': 'failed',
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'message': f'Images differ (max_diff={max_diff:.2f}, mean_diff={mean_diff:.4f})'
            })
            results['failed'] += 1
    
    return results

def print_results(results: Dict):
    """Print detailed results for an implementation"""
    impl_name = results['implementation']
    total = results['total']
    passed = results['passed']
    failed = results['failed']
    missing = results['missing']
    
    # Determine overall status color
    if missing == total:
        status_color = Colors.FAIL
        status = "NO OUTPUT FOUND"
    elif passed == total:
        status_color = Colors.OKGREEN
        status = "ALL PASSED"
    elif passed > 0:
        status_color = Colors.WARNING
        status = f"{passed}/{total} PASSED"
    else:
        status_color = Colors.FAIL
        status = "ALL FAILED"
    
    # Print summary
    print(f"\n{Colors.BOLD}{impl_name}:{Colors.ENDC} {status_color}{status}{Colors.ENDC}")
    
    if missing == total:
        print(f"  {Colors.FAIL}⚠  No output files were found. Run the implementation first.{Colors.ENDC}")
        return
    
    # Print detailed results
    for detail in results['details']:
        file_name = detail['file']
        status = detail['status']
        
        if status == 'passed':
            print(f"  {Colors.OKGREEN}✓{Colors.ENDC} {file_name}: {Colors.OKGREEN}IDENTICAL{Colors.ENDC}")
        elif status == 'failed':
            max_diff = detail.get('max_diff', 0)
            mean_diff = detail.get('mean_diff', 0)
            print(f"  {Colors.FAIL}✗{Colors.ENDC} {file_name}: {Colors.FAIL}DIFFERENT{Colors.ENDC} "
                  f"(max_diff={max_diff:.2f}, mean_diff={mean_diff:.4f})")
        elif status == 'gpu_missing':
            print(f"  {Colors.WARNING}⚠{Colors.ENDC} {file_name}: {Colors.WARNING}MISSING (not generated by GPU){Colors.ENDC}")
        elif status == 'cpu_missing':
            print(f"  {Colors.WARNING}⚠{Colors.ENDC} {file_name}: {Colors.WARNING}MISSING (ground truth not found){Colors.ENDC}")
    
    # Print statistics
    if passed > 0 or failed > 0:
        print(f"\n  Statistics:")
        print(f"    Passed:  {Colors.OKGREEN}{passed}/{total}{Colors.ENDC}")
        print(f"    Failed:  {Colors.FAIL}{failed}/{total}{Colors.ENDC}")
        if missing > 0:
            print(f"    Missing: {Colors.WARNING}{missing}/{total}{Colors.ENDC}")

def main():
    """Main function to run all comparisons"""
    print_header("GPU Implementation Validation Report")
    
    # Define base directories
    base_dir = Path(__file__).parent.parent.parent
    cpu_dir = base_dir / "CPU" / "image_addition"
    gpu_base_dir = base_dir / "GPU" / "image_addition"
    
    # Expected output files (based on the code analysis)
    expected_outputs = [
        "out_average.jpg",
        "out_constant.jpg",
        "out_gradient.jpg",
        "out_grad_avg.jpg",
        "out_grad_sat.jpg",
        "out_blend.jpg"
    ]
    
    # CUDA C++ uses different naming convention
    cuda_cpp_outputs = [
        "cuda_out_1_avg.jpg",
        "cuda_out_2_const.jpg",
        "cuda_out_3_grad.jpg",
        "cuda_out_4_grad_avg.jpg",
        "cuda_out_5_grad_sat.jpg",
        "cuda_out_6_blend.jpg"
    ]
    
    cuda_cpp_outputs = [
        "cuda_out_1_avg.jpg",
        "cuda_out_2_const.jpg",
        "cuda_out_3_grad.jpg",
        "cuda_out_4_grad_avg.jpg",
        "cuda_out_5_grad_sat.jpg",
        "cuda_out_6_blend.jpg"
    ]
    
    # Check if ground truth exists
    print_section("Checking Ground Truth (CPU Implementation)")
    ground_truth_exists = all((cpu_dir / f).exists() for f in expected_outputs)
    
    if ground_truth_exists:
        print(f"{Colors.OKGREEN}✓ All ground truth files found{Colors.ENDC}")
        for output_file in expected_outputs:
            cpu_path = cpu_dir / output_file
            file_size = cpu_path.stat().st_size / 1024  # KB
            print(f"  • {output_file}: {file_size:.1f} KB")
    else:
        print(f"{Colors.FAIL}⚠ Some ground truth files are missing!{Colors.ENDC}")
        for output_file in expected_outputs:
            cpu_path = cpu_dir / output_file
            if cpu_path.exists():
                print(f"  {Colors.OKGREEN}✓{Colors.ENDC} {output_file}")
            else:
                print(f"  {Colors.FAIL}✗{Colors.ENDC} {output_file}")
        print(f"\n{Colors.WARNING}Run the CPU implementation first to generate ground truth.{Colors.ENDC}")
        return
    
    # Check GPU implementations
    print_section("Validating GPU Implementations")
    
    implementations = [
        ("OpenACC", gpu_base_dir / "openACC", expected_outputs, None),
        ("CUDA Python", gpu_base_dir / "CUDA_Python", expected_outputs, None),
        ("CUDA C++", gpu_base_dir / "CUDA_CPP", cuda_cpp_outputs, None),
    ]
    
    all_results = []
    
    for impl_name, impl_dir, outputs, mapping in implementations:
        if not impl_dir.exists():
            print(f"\n{Colors.WARNING}⚠ {impl_name} directory not found: {impl_dir}{Colors.ENDC}")
            continue
        
        results = check_implementation(impl_name, impl_dir, cpu_dir, outputs, mapping)
        all_results.append(results)
        print_results(results)
    
    # Print final summary
    print_section("Final Summary")
    
    for results in all_results:
        impl_name = results['implementation']
        total = results['total']
        passed = results['passed']
        missing = results['missing']
        
        if missing == total:
            print(f"{Colors.FAIL}✗ {impl_name}: NO OUTPUT (run implementation first){Colors.ENDC}")
        elif passed == total:
            print(f"{Colors.OKGREEN}✓ {impl_name}: {passed}/{total} - ALL TESTS PASSED{Colors.ENDC}")
        elif passed > 0:
            print(f"{Colors.WARNING}⚠ {impl_name}: {passed}/{total} - PARTIAL PASS{Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}✗ {impl_name}: {passed}/{total} - ALL TESTS FAILED{Colors.ENDC}")
    
    print(f"\n{Colors.BOLD}Validation complete.{Colors.ENDC}\n")

if __name__ == "__main__":
    main()
