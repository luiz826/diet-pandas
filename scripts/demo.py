#!/usr/bin/env python3
"""
Diet Pandas - Quick Demo

This script provides a quick demonstration of Diet Pandas capabilities.
Run this to see the library in action!
"""

import pandas as pd
import numpy as np


def print_banner(text):
    """Print a styled banner."""
    width = 80
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width + "\n")


def demo():
    """Run a quick demo of Diet Pandas."""
    
    print_banner("Diet Pandas - Quick Demo")
    
    # Step 1: Create sample data
    print("Creating sample DataFrame with wasteful types...")
    
    np.random.seed(42)
    df = pd.DataFrame({
        'user_id': np.arange(1000, 6000, dtype='int64'),
        'age': np.random.randint(18, 80, size=5000, dtype='int64'),
        'score': np.random.uniform(0, 100, size=5000).astype('float64'),
        'category': np.random.choice(['A', 'B', 'C', 'D'], size=5000),
        'country': np.random.choice(['USA', 'UK', 'Canada', 'Germany'], size=5000),
    })
    
    print(f"\nDataFrame created: {len(df):,} rows, {len(df.columns)} columns")
    print("\nData types BEFORE optimization:")
    print(df.dtypes)
    
    original_memory = df.memory_usage(deep=True).sum()
    print(f"\nMemory usage BEFORE: {original_memory / 1e6:.2f} MB")
    
    # Step 2: Apply Diet
    print_banner("Applying Diet Pandas Optimization")
    
    try:
        import dietpandas as dp
        
        df_optimized = dp.diet(df, verbose=True)
        
        print("\nData types AFTER optimization:")
        print(df_optimized.dtypes)
        
        optimized_memory = df_optimized.memory_usage(deep=True).sum()
        print(f"\nMemory usage AFTER: {optimized_memory / 1e6:.2f} MB")
        
        # Step 3: Verify data integrity
        print_banner("Verifying Data Integrity")
        
        print("Checking that data values are preserved...")
        
        checks = {
            'user_id preserved': (df['user_id'] == df_optimized['user_id']).all(),
            'age preserved': (df['age'] == df_optimized['age']).all(),
            'score preserved (approx)': np.allclose(df['score'], df_optimized['score']),
            'category preserved': (df['category'] == df_optimized['category']).all(),
            'country preserved': (df['country'] == df_optimized['country']).all(),
        }
        
        for check, result in checks.items():
            status = "[PASS]" if result else "[FAIL]"
            print(f"{status} {check}")
        
        # Step 4: Show detailed report
        print_banner("Detailed Memory Report")
        
        report = dp.get_memory_report(df_optimized)
        print(report.to_string(index=False))
        
        # Step 5: Summary
        print_banner("Demo Complete!")
        
        reduction = 100 * (original_memory - optimized_memory) / original_memory
        print(f"Memory reduced by {reduction:.1f}%")
        print(f"{original_memory/1e6:.2f} MB -> {optimized_memory/1e6:.2f} MB")
        print(f"Saved {(original_memory - optimized_memory)/1e6:.2f} MB\n")
        
        print("Ready to use Diet Pandas!")
        print("\nNext steps:")
        print("  - Read the docs: cat README.md")
        print("  - See examples: python scripts/examples.py")
        print("  - Run tests: pytest tests/ -v\n")
        
    except ImportError:
        print("[ERROR] Diet Pandas not installed!")
        print("\nPlease install first:")
        print("  pip install -e .")
        print("\nOr run the setup script:")
        print("  python scripts/quickstart.py")
        return
    
    except Exception as e:
        print(f"[ERROR] Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    demo()
