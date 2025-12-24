"""
NYC Taxi Data Benchmark
========================

Benchmarks diet-pandas optimization on NYC Yellow Taxi Trip Records.
Tests both individual monthly files and a combined multi-month dataset.

Data source: NYC TLC Trip Record Data
Files tested:
- yellow_tripdata_2015-01.csv (~12.7M rows, 1.8GB)
- yellow_tripdata_2016-01.csv (~10.9M rows, 1.6GB)
- yellow_tripdata_2016-02.csv (~11.4M rows, 1.7GB)
- yellow_tripdata_2016-03.csv (~12.2M rows, 1.8GB)
- Combined dataset (~47M rows, ~7GB)
"""

import os
import sys
import time
import json
from pathlib import Path

import pandas as pd
import psutil

# Add dietpandas to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
import dietpandas as dp


def get_memory_usage():
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def benchmark_file(file_path: str, output_name: str):
    """Benchmark a single taxi CSV file."""
    print(f"\n{'='*80}")
    print(f"Benchmarking: {output_name}")
    print(f"File: {file_path}")
    print(f"Size: {os.path.getsize(file_path) / (1024**3):.2f} GB")
    print(f"{'='*80}\n")
    
    results = {
        'file': output_name,
        'file_path': file_path,
        'file_size_gb': round(os.path.getsize(file_path) / (1024**3), 2)
    }
    
    # Benchmark standard pandas
    print("📊 Standard pandas.read_csv()...")
    mem_before = get_memory_usage()
    start_time = time.time()
    
    df_standard = pd.read_csv(file_path)
    
    load_time_standard = time.time() - start_time
    mem_after = get_memory_usage()
    mem_used_standard = df_standard.memory_usage(deep=True).sum() / (1024 * 1024)
    
    results['standard'] = {
        'rows': len(df_standard),
        'columns': len(df_standard.columns),
        'memory_mb': round(mem_used_standard, 2),
        'load_time_seconds': round(load_time_standard, 2),
        'process_memory_mb': round(mem_after - mem_before, 2)
    }
    
    print(f"   Rows: {len(df_standard):,}")
    print(f"   Memory: {mem_used_standard:,.2f} MB")
    print(f"   Load time: {load_time_standard:.2f}s")
    
    # Get column dtypes for analysis
    print("\n📋 Column types:")
    dtype_counts = df_standard.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {dtype}: {count} columns")
    
    # Benchmark diet-pandas
    print("\n🥗 diet-pandas optimization...")
    mem_before = get_memory_usage()
    start_time = time.time()
    
    df_optimized = dp.read_csv(file_path)
    
    load_time_optimized = time.time() - start_time
    mem_after = get_memory_usage()
    mem_used_optimized = df_optimized.memory_usage(deep=True).sum() / (1024 * 1024)
    
    results['optimized'] = {
        'rows': len(df_optimized),
        'columns': len(df_optimized.columns),
        'memory_mb': round(mem_used_optimized, 2),
        'load_time_seconds': round(load_time_optimized, 2),
        'process_memory_mb': round(mem_after - mem_before, 2)
    }
    
    print(f"   Rows: {len(df_optimized):,}")
    print(f"   Memory: {mem_used_optimized:,.2f} MB")
    print(f"   Load time: {load_time_optimized:.2f}s")
    
    # Calculate improvements
    memory_saved = mem_used_standard - mem_used_optimized
    memory_reduction_pct = (memory_saved / mem_used_standard) * 100
    time_overhead_pct = ((load_time_optimized - load_time_standard) / load_time_standard) * 100
    
    results['improvement'] = {
        'memory_saved_mb': round(memory_saved, 2),
        'memory_reduction_percent': round(memory_reduction_pct, 2),
        'time_overhead_percent': round(time_overhead_pct, 2)
    }
    
    print(f"\n✨ Results:")
    print(f"   Memory saved: {memory_saved:,.2f} MB ({memory_reduction_pct:.1f}% reduction)")
    print(f"   Time overhead: {time_overhead_pct:.1f}%")
    
    # Column-by-column comparison
    print(f"\n📊 Column-by-column optimization:")
    memory_by_column = []
    for col in df_standard.columns:
        mem_std = df_standard[col].memory_usage(deep=True) / (1024 * 1024)
        mem_opt = df_optimized[col].memory_usage(deep=True) / (1024 * 1024)
        saved = mem_std - mem_opt
        if saved > 0.1:  # Only show significant savings
            reduction = (saved / mem_std) * 100
            memory_by_column.append({
                'column': col,
                'standard_mb': round(mem_std, 2),
                'optimized_mb': round(mem_opt, 2),
                'saved_mb': round(saved, 2),
                'reduction_percent': round(reduction, 2)
            })
            print(f"   {col:30s}: {mem_std:8.2f} MB → {mem_opt:8.2f} MB ({reduction:5.1f}% reduction)")
    
    results['column_optimization'] = memory_by_column
    
    # Clean up
    del df_standard, df_optimized
    
    return results


def create_combined_file(taxi_dir: Path, output_path: Path):
    """Combine all taxi CSV files into one huge file."""
    print(f"\n{'='*80}")
    print("Creating combined taxi dataset...")
    print(f"{'='*80}\n")
    
    taxi_files = sorted(taxi_dir.glob('yellow_tripdata_*.csv'))
    print(f"Found {len(taxi_files)} files to combine:")
    for f in taxi_files:
        print(f"  - {f.name} ({os.path.getsize(f) / (1024**3):.2f} GB)")
    
    print(f"\nCombining into: {output_path}")
    print("This may take several minutes...\n")
    
    start_time = time.time()
    
    # Read first file to get header
    first_df = pd.read_csv(taxi_files[0], nrows=1)
    header = first_df.columns.tolist()
    
    # Write combined file
    with open(output_path, 'w') as outfile:
        # Write header
        outfile.write(','.join(header) + '\n')
        
        # Append all files
        for i, file_path in enumerate(taxi_files, 1):
            print(f"  [{i}/{len(taxi_files)}] Processing {file_path.name}...")
            
            # Read in chunks to manage memory
            chunk_size = 100000
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                chunk.to_csv(outfile, header=False, index=False)
    
    elapsed = time.time() - start_time
    combined_size = os.path.getsize(output_path) / (1024**3)
    
    print(f"\n✅ Combined file created!")
    print(f"   Size: {combined_size:.2f} GB")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Path: {output_path}")
    
    return output_path


def main():
    """Run taxi benchmark suite."""
    print("🚖 NYC Taxi Data Benchmark")
    print("=" * 80)
    print("Testing diet-pandas optimization on real-world taxi trip records\n")
    
    # Setup paths
    taxi_dir = Path('/Users/luizfernando/dev/bench-diet-pandas/taxi')
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    all_results = {
        'dataset': 'NYC Yellow Taxi Trip Records',
        'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'individual_files': [],
        'combined_file': None
    }
    
    # Benchmark individual files
    taxi_files = [
        ('yellow_tripdata_2015-01.csv', '2015-01 (January 2015)'),
        ('yellow_tripdata_2016-01.csv', '2016-01 (January 2016)'),
        ('yellow_tripdata_2016-02.csv', '2016-02 (February 2016)'),
        ('yellow_tripdata_2016-03.csv', '2016-03 (March 2016)'),
    ]
    
    print("\n🔍 Part 1: Individual Monthly Files")
    print("=" * 80)
    
    for filename, label in taxi_files:
        file_path = taxi_dir / filename
        if file_path.exists():
            results = benchmark_file(str(file_path), label)
            all_results['individual_files'].append(results)
        else:
            print(f"⚠️  File not found: {file_path}")
    
    # Create and benchmark combined file
    print("\n\n🔍 Part 2: Combined Multi-Month Dataset")
    print("=" * 80)
    
    combined_file = taxi_dir / 'yellow_tripdata_combined.csv'
    
    if not combined_file.exists():
        print("Combined file not found. Creating it now...")
        create_combined_file(taxi_dir, combined_file)
    else:
        print(f"Using existing combined file: {combined_file}")
        print(f"Size: {os.path.getsize(combined_file) / (1024**3):.2f} GB")
    
    if combined_file.exists():
        combined_results = benchmark_file(
            str(combined_file),
            'Combined Dataset (2015-01 + 2016 Q1)'
        )
        all_results['combined_file'] = combined_results
    
    # Save results
    json_path = results_dir / 'taxi_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\n📊 Results saved to: {json_path}")
    
    # Generate markdown report
    generate_report(all_results, results_dir / 'taxi_results.md')
    
    print(f"\n{'='*80}")
    print("✅ Benchmark complete!")
    print(f"{'='*80}\n")


def generate_report(results: dict, output_path: Path):
    """Generate markdown report from benchmark results."""
    with open(output_path, 'w') as f:
        f.write("# NYC Taxi Data Benchmark Results\n\n")
        f.write(f"**Date:** {results['date']}\n\n")
        f.write(f"**Dataset:** {results['dataset']}\n\n")
        
        # Individual files summary
        if results['individual_files']:
            f.write("## Individual Monthly Files\n\n")
            f.write("| File | Rows | File Size | Standard | Optimized | Saved | Reduction |\n")
            f.write("|------|------|-----------|----------|-----------|-------|----------|\n")
            
            for r in results['individual_files']:
                f.write(f"| {r['file']} | {r['standard']['rows']:,} | "
                       f"{r['file_size_gb']:.2f} GB | "
                       f"{r['standard']['memory_mb']:,.0f} MB | "
                       f"{r['optimized']['memory_mb']:,.0f} MB | "
                       f"{r['improvement']['memory_saved_mb']:,.0f} MB | "
                       f"**{r['improvement']['memory_reduction_percent']:.1f}%** |\n")
            
            f.write("\n### Performance Details\n\n")
            
            for r in results['individual_files']:
                f.write(f"#### {r['file']}\n\n")
                f.write(f"- **Rows:** {r['standard']['rows']:,}\n")
                f.write(f"- **Columns:** {r['standard']['columns']}\n")
                f.write(f"- **File size:** {r['file_size_gb']:.2f} GB\n\n")
                
                f.write("**Memory Usage:**\n")
                f.write(f"- Standard: {r['standard']['memory_mb']:,.2f} MB\n")
                f.write(f"- Optimized: {r['optimized']['memory_mb']:,.2f} MB\n")
                f.write(f"- **Saved: {r['improvement']['memory_saved_mb']:,.2f} MB "
                       f"({r['improvement']['memory_reduction_percent']:.1f}% reduction)**\n\n")
                
                f.write("**Load Time:**\n")
                f.write(f"- Standard: {r['standard']['load_time_seconds']:.2f}s\n")
                f.write(f"- Optimized: {r['optimized']['load_time_seconds']:.2f}s\n")
                f.write(f"- Overhead: {r['improvement']['time_overhead_percent']:.1f}%\n\n")
                
                if r.get('column_optimization'):
                    f.write("**Top Column Optimizations:**\n\n")
                    sorted_cols = sorted(r['column_optimization'], 
                                       key=lambda x: x['saved_mb'], 
                                       reverse=True)[:10]
                    for col in sorted_cols:
                        f.write(f"- `{col['column']}`: {col['saved_mb']:.2f} MB saved "
                               f"({col['reduction_percent']:.1f}% reduction)\n")
                    f.write("\n")
        
        # Combined file results
        if results['combined_file']:
            r = results['combined_file']
            f.write("## Combined Multi-Month Dataset\n\n")
            f.write(f"**File:** {r['file']}\n\n")
            f.write(f"- **Total rows:** {r['standard']['rows']:,}\n")
            f.write(f"- **Columns:** {r['standard']['columns']}\n")
            f.write(f"- **File size:** {r['file_size_gb']:.2f} GB\n\n")
            
            f.write("### Memory Usage\n\n")
            f.write(f"- **Standard pandas:** {r['standard']['memory_mb']:,.0f} MB\n")
            f.write(f"- **diet-pandas:** {r['optimized']['memory_mb']:,.0f} MB\n")
            f.write(f"- **Saved:** {r['improvement']['memory_saved_mb']:,.0f} MB "
                   f"({r['improvement']['memory_reduction_percent']:.1f}% reduction)\n\n")
            
            f.write("### Load Time\n\n")
            f.write(f"- **Standard:** {r['standard']['load_time_seconds']:.2f}s\n")
            f.write(f"- **Optimized:** {r['optimized']['load_time_seconds']:.2f}s\n")
            f.write(f"- **Overhead:** {r['improvement']['time_overhead_percent']:.1f}%\n\n")
            
            if r.get('column_optimization'):
                f.write("### Top Column Optimizations\n\n")
                sorted_cols = sorted(r['column_optimization'], 
                                   key=lambda x: x['saved_mb'], 
                                   reverse=True)[:15]
                f.write("| Column | Standard | Optimized | Saved | Reduction |\n")
                f.write("|--------|----------|-----------|-------|----------|\n")
                for col in sorted_cols:
                    f.write(f"| `{col['column']}` | {col['standard_mb']:.2f} MB | "
                           f"{col['optimized_mb']:.2f} MB | {col['saved_mb']:.2f} MB | "
                           f"{col['reduction_percent']:.1f}% |\n")
        
        f.write("\n---\n\n")
        f.write("*Benchmark generated by diet-pandas*\n")
    
    print(f"📄 Markdown report saved to: {output_path}")


if __name__ == '__main__':
    main()
