"""
Real-World Benchmark: ENEM 2024 Dataset
Brazilian National Exam - 4.5 million participants
"""

import time
import pandas as pd
import sys
from pathlib import Path

# Add diet-pandas to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import dietpandas as dp


def measure_memory(df):
    """Return DataFrame memory usage in MB"""
    return df.memory_usage(deep=True).sum() / 1024**2


def benchmark_file(file_path, file_name):
    """Benchmark a single ENEM CSV file"""
    print(f"\n{'=' * 70}")
    print(f"BENCHMARKING: {file_name}")
    print(f"{'=' * 70}")

    # Get file size
    file_size_mb = file_path.stat().st_size / 1024**2
    print(f"\nFile size: {file_size_mb:.2f} MB")

    # Benchmark 1: Standard pandas
    print(f"\n1️⃣  Standard pandas.read_csv()...")
    start_time = time.time()
    df_pandas = pd.read_csv(file_path, sep=";", encoding="latin1", low_memory=False)
    pandas_load_time = time.time() - start_time

    pandas_memory = measure_memory(df_pandas)

    print(f"   ✓ Load time:  {pandas_load_time:.2f} seconds")
    print(f"   ✓ Memory:     {pandas_memory:.2f} MB")
    print(f"   ✓ Shape:      {df_pandas.shape[0]:,} rows × {df_pandas.shape[1]} columns")

    # Benchmark 2: Diet-pandas read_csv (with auto-optimization)
    print(f"\n2️⃣  Diet-pandas dp.read_csv() with auto-optimization...")
    start_time = time.time()
    df_diet = dp.read_csv(file_path, sep=";", encoding="latin1", verbose=False)
    diet_load_time = time.time() - start_time

    diet_memory = measure_memory(df_diet)

    print(f"   ✓ Load time:  {diet_load_time:.2f} seconds")
    print(f"   ✓ Memory:     {diet_memory:.2f} MB")
    print(f"   ✓ Shape:      {df_diet.shape[0]:,} rows × {df_diet.shape[1]} columns")

    # Calculate improvements
    memory_reduction = ((pandas_memory - diet_memory) / pandas_memory) * 100
    memory_saved = pandas_memory - diet_memory

    # Results
    print(f"\n{'=' * 70}")
    print(f"📊 RESULTS FOR {file_name}")
    print(f"{'=' * 70}")
    print(f"\n💾 Memory:")
    print(f"   • Pandas:       {pandas_memory:>10.2f} MB")
    print(f"   • Diet-Pandas:  {diet_memory:>10.2f} MB")
    print(f"   • Reduction:    {memory_reduction:>10.1f}%")
    print(f"   • Saved:        {memory_saved:>10.2f} MB")

    print(f"\n⏱️  Load Time:")
    print(f"   • Pandas:       {pandas_load_time:>10.2f} sec")
    print(f"   • Diet-Pandas:  {diet_load_time:>10.2f} sec")

    if diet_load_time < pandas_load_time:
        speedup = pandas_load_time / diet_load_time
        print(f"   • Speedup:      {speedup:>10.2f}x faster ⚡")
    else:
        slowdown = diet_load_time / pandas_load_time
        print(
            f"   • Trade-off:    {slowdown:>10.2f}x slower (but saves {memory_reduction:.0f}% RAM!)"
        )

    # Show sample optimizations
    print(f"\n🔍 Sample Column Optimizations:")
    print(f"   {'Column':<25} {'Pandas':<15} → {'Diet':<15}")
    print(f"   {'-'*60}")
    for i, col in enumerate(df_pandas.columns[:10]):  # First 10 columns
        pandas_dtype = str(df_pandas[col].dtype)
        diet_dtype = str(df_diet[col].dtype)
        if pandas_dtype != diet_dtype:
            print(f"   {col:<25} {pandas_dtype:<15} → {diet_dtype:<15}")

    return {
        "file_name": file_name,
        "file_size_mb": file_size_mb,
        "rows": df_pandas.shape[0],
        "columns": df_pandas.shape[1],
        "pandas_memory_mb": pandas_memory,
        "diet_memory_mb": diet_memory,
        "memory_reduction_pct": memory_reduction,
        "memory_saved_mb": memory_saved,
        "pandas_load_time_sec": pandas_load_time,
        "diet_load_time_sec": diet_load_time,
    }


def main():
    """Run ENEM benchmarks"""
    print("\n" + "=" * 70)
    print("🇧🇷 ENEM 2024 REAL-WORLD BENCHMARK")
    print("=" * 70)
    print("\nDataset: Brazilian National Exam (ENEM) 2024")
    print("Source: INEP - Ministry of Education")

    # Path to ENEM data
    data_path = Path(__file__).parent.parent.parent / "bench-diet-pandas" / "microdados_enem_2024" / "DADOS"

    if not data_path.exists():
        print(f"\n❌ ERROR: Data directory not found: {data_path}")
        print("Please ensure ENEM data is available at the expected location.")
        return

    # Files to benchmark
    files_to_test = [
        ("RESULTADOS_2024.csv", "Exam Results - Main file"),
        ("PARTICIPANTES_2024.csv", "Participants Demographics"),
    ]

    results = []

    for filename, description in files_to_test:
        file_path = data_path / filename

        if not file_path.exists():
            print(f"\n⚠️  Skipping {filename} - file not found")
            continue

        print(f"\n📁 {description}")
        result = benchmark_file(file_path, filename)
        results.append(result)

    # Summary table
    if results:
        print("\n" + "=" * 70)
        print("📋 SUMMARY TABLE")
        print("=" * 70)
        print(
            f"\n{'File':<30} {'Rows':<12} {'Memory Reduction':<18} {'Load Time':<15}"
        )
        print("-" * 75)

        for r in results:
            print(
                f"{r['file_name']:<30} {r['rows']:>10,}  "
                f"{r['memory_reduction_pct']:>8.1f}%         "
                f"{r['pandas_load_time_sec']:>6.2f}s → {r['diet_load_time_sec']:>6.2f}s"
            )

        # Save results
        import json

        output_path = Path(__file__).parent / "results" / "enem_results.json"
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to: {output_path}")

    print("\n" + "=" * 70)
    print("✨ BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
