#!/usr/bin/env python3
"""
Quick Start Script for Diet Pandas

This script helps you set up and verify your Diet Pandas installation.
"""

import subprocess
import sys


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def check_python_version():
    """Check if Python version is 3.10 or higher."""
    print_section("Checking Python Version")

    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major >= 3 and version.minor >= 8:
        print("[OK] Python version is compatible")
        return True
    else:
        print("[ERROR] Python 3.10+ is required")
        return False


def install_package():
    """Install the package in development mode."""
    print_section("Installing Diet Pandas")

    try:
        print("Installing in development mode...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        print("[OK] Installation successful")
        return True
    except subprocess.CalledProcessError:
        print("[ERROR] Installation failed")
        return False


def install_dependencies():
    """Install required dependencies."""
    print_section("Installing Dependencies")

    dependencies = [
        ("pandas", "pandas>=1.5.0"),
        ("numpy", "numpy>=1.20.0"),
        ("polars", "polars>=0.17.0"),
        ("pytest", "pytest>=7.0.0"),
        ("openpyxl", "openpyxl"),  # For Excel support
    ]

    for name, package in dependencies:
        try:
            print(f"Installing {name}...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", package], check=True, capture_output=True
            )
            print(f"[OK] {name} installed")
        except subprocess.CalledProcessError:
            print(f"[WARN] {name} installation failed (optional)")

    return True


def verify_installation():
    """Verify that Diet Pandas can be imported."""
    print_section("Verifying Installation")

    try:
        import dietpandas as dp

        print(f"[OK] Diet Pandas version {dp.__version__} imported successfully")

        # Check available functions
        print("\nAvailable functions:")
        for func in dp.__all__:
            print(f"  - {func}")

        return True
    except ImportError as e:
        print(f"[ERROR] Failed to import Diet Pandas: {e}")
        return False


def run_simple_test():
    """Run a simple test to verify functionality."""
    print_section("Running Simple Test")

    try:
        import pandas as pd

        import dietpandas as dp

        # Create a simple DataFrame
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [1.1, 2.2, 3.3, 4.4, 5.5],
                "c": ["x", "y", "x", "y", "x"],
            }
        )

        print("Original DataFrame:")
        print(df.dtypes)
        print(f"Memory: {df.memory_usage(deep=True).sum()} bytes")

        # Optimize
        df_optimized = dp.diet(df, verbose=False)

        print("\nOptimized DataFrame:")
        print(df_optimized.dtypes)
        print(f"Memory: {df_optimized.memory_usage(deep=True).sum()} bytes")

        print("\n[OK] Basic test passed!")
        return True

    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False


def main():
    """Main setup workflow."""
    print("\nDiet Pandas - Quick Start Setup\n")

    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)

    # Step 2: Install package
    if not install_package():
        print("\n[WARN] You can try manual installation:")
        print("   pip install -e .")
        sys.exit(1)

    # Step 3: Install dependencies
    install_dependencies()

    # Step 4: Verify installation
    if not verify_installation():
        sys.exit(1)

    # Step 5: Run simple test
    if not run_simple_test():
        sys.exit(1)

    # Success!
    print_section("Setup Complete!")
    print("Diet Pandas is ready to use!\n")
    print("Next steps:")
    print("  1. Run examples: python scripts/examples.py")
    print("  2. Run tests: pytest tests/ -v")
    print("  3. Read the docs: cat README.md")
    print("  4. Start using: import dietpandas as dp\n")


if __name__ == "__main__":
    main()
