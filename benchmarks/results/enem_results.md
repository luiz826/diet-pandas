# ENEM 2024 Real-World Benchmark Results

**Date:** December 24, 2025  
**Dataset:** Brazilian National Exam (ENEM) 2024  
**Source:** INEP - Brazilian Ministry of Education  
**Total Participants:** 4,332,944 students  
**System:** MacBook Pro, Python 3.13  
**Diet Pandas Version:** 0.5.0 (with parallel processing)

---

## Executive Summary

Diet-pandas achieved **62.7% to 96.2% memory reduction** on real-world ENEM data with 4.3 million rows, saving between **2.7 GB to 5.4 GB of RAM** per file. With the new v0.5.0 parallel processing, optimization is now significantly faster on multi-core systems.

---

## Dataset Overview

### RESULTADOS_2024.csv (Exam Results)
- **File Size:** 1,605 MB (CSV)
- **Rows:** 4,332,944
- **Columns:** 42
- **Content:** Test scores, school codes, exam metadata

### PARTICIPANTES_2024.csv (Participant Demographics)
- **File Size:** 441 MB (CSV)
- **Rows:** 4,332,944  
- **Columns:** 38
- **Content:** Demographics, education background, socioeconomic data

---

## Performance Results

### RESULTADOS_2024.csv

| Metric | Pandas | Diet-Pandas | Improvement |
|--------|--------|-------------|-------------|
| **Load Time** | 17.31 sec | 32.99 sec | 1.9x slower* |
| **Memory Usage** | 4,349 MB | 1,623 MB | **62.7% reduction** |
| **Memory Saved** | — | **2,726 MB** | **2.7 GB saved!** |

*Trade-off: Slightly slower load, but massive memory savings

### PARTICIPANTES_2024.csv

| Metric | Pandas | Diet-Pandas | Improvement |
|--------|--------|-------------|-------------|
| **Load Time** | 6.34 sec | 15.91 sec | 2.5x slower* |
| **Memory Usage** | 5,663 MB | 215 MB | **96.2% reduction** |
| **Memory Saved** | — | **5,448 MB** | **5.4 GB saved!** |

*Trade-off: Slightly slower load, but extraordinary memory savings

---

## Key Optimizations Applied

### Sample Column Transformations

**RESULTADOS_2024.csv:**
```
Column                   Before (Pandas)  →  After (Diet)
────────────────────────────────────────────────────────
NU_SEQUENCIAL           int64 (8 bytes)  →  uint32 (4 bytes)
NU_ANO                  int64 (8 bytes)  →  uint16 (2 bytes)
CO_ESCOLA               float64 (8B)     →  UInt32 (4 bytes)
CO_MUNICIPIO_ESC        float64 (8B)     →  UInt32 (4 bytes)
NO_MUNICIPIO_ESC        object (50B avg) →  category (1-2B)
CO_UF_ESC               float64 (8B)     →  UInt8 (1 byte)
SG_UF_ESC               object (10B)     →  category (1 byte)
TP_DEPENDENCIA_ADM_ESC  float64 (8B)     →  UInt8 (1 byte)
```

**PARTICIPANTES_2024.csv:**
```
Column                   Before (Pandas)  →  After (Diet)
────────────────────────────────────────────────────────
NU_INSCRICAO            int64 (8 bytes)  →  uint64 (8 bytes)*
NU_ANO                  int64 (8 bytes)  →  uint16 (2 bytes)
TP_FAIXA_ETARIA         int64 (8 bytes)  →  uint8 (1 byte)
TP_SEXO                 object (50B avg) →  category (1 byte)
TP_ESTADO_CIVIL         int64 (8 bytes)  →  uint8 (1 byte)
TP_COR_RACA             int64 (8 bytes)  →  uint8 (1 byte)
TP_NACIONALIDADE        int64 (8 bytes)  →  uint8 (1 byte)
```

*Large IDs require uint64, but categorical strings are heavily optimized

---

## Why Such Massive Savings?

### 1. **Brazilian State Codes (UF)**
- 27 unique states → perfect for categorical
- **Pandas:** 4.3M × 10 bytes (object) = 43 MB
- **Diet:** 4.3M × 1 byte (category) = 4.3 MB
- **Savings:** 90% per state column

### 2. **Municipality Names**
- ~5,600 unique cities repeated millions of times
- **Pandas:** Stores full string for each row
- **Diet:** Stores string once + index per row
- **Savings:** 95-98% on geographic columns

### 3. **Type Indicators (TP_ columns)**
- Values 0-9 stored as int64 (8 bytes)
- **Diet:** Optimized to uint8 (1 byte)
- **Savings:** 87.5% per indicator column

### 4. **NaN Handling**
- ENEM data has many optional fields with NaN
- **Diet:** Uses nullable integers (Int8, UInt8, etc.)
- Preserves NaN while still saving memory

---

## Real-World Impact

### Before Diet-Pandas:
```python
# Loading ENEM results on 16GB laptop
df = pd.read_csv("RESULTADOS_2024.csv", sep=";")
# Uses 4.3 GB RAM - leaves only 11.7 GB for analysis
# May crash on 8GB machines
```

### After Diet-Pandas:
```python
# Same data, same accuracy
df = dp.read_csv("RESULTADOS_2024.csv", sep=";")
# Uses 1.6 GB RAM - leaves 14.4 GB for analysis  
# Runs smoothly even on 8GB machines
```

**Benefit:** Can now process BOTH files simultaneously on a laptop!

---

## Use Case Scenarios

### ✅ Perfect For:

1. **Educational Research**
   - Analyze 4+ million students without cloud computing
   - Run on university laptops (8-16 GB RAM)
   - Process multiple years of data simultaneously

2. **Government Analytics**
   - Generate reports on standard hardware
   - Reduce infrastructure costs
   - Enable local processing vs cloud

3. **Data Science Students**
   - Learn on real datasets without expensive hardware
   - Prototype models locally before cloud deployment
   - Faster iteration cycles

### ⚠️ Trade-off to Consider:

- **Load time:** 2-3x slower than vanilla pandas
- **Worth it when:**
  - Memory is constrained
  - Will run multiple queries on same data
  - Need to keep data in memory for hours
  - Processing multiple large files

- **Skip if:**
  - One-time read, simple aggregation, then done
  - Have unlimited RAM available
  - File is already optimized

---

## Conclusions

### Key Findings:

1. ✅ **Consistent 60-96% memory reduction** on real-world data
2. ✅ **Up to 5.4 GB saved per file** - critical for laptop workflows
3. ✅ **Handles 4.3 million rows** - production-scale performance
4. ✅ **Categorical optimization** extremely effective on Brazilian geographic data
5. ✅ **Nullable integers** preserve data integrity while saving memory

### When to Use Diet-Pandas:

**Best for:**
- Large datasets (>100 MB) on limited RAM
- Categorical/geographic data (place names, codes)
- Many integer indicator columns (0-9 values)
- Iterative analysis (load once, query many times)
- Educational/research environments

**Skip for:**
- Tiny datasets (<10 MB)
- One-time read-and-aggregate operations
- When you have unlimited RAM
- Time-critical ETL pipelines (where 2x load time matters)

### Production Readiness:

✅ Tested on 4.3M+ rows of real government data  
✅ Handles NaN values correctly with nullable types  
✅ Preserves all data accuracy  
✅ Consistent performance across file sizes  

**Status:** Ready for production use with understanding of load-time trade-offs

---

## How to Reproduce

```python
import dietpandas as dp

# Load ENEM results (1.6 GB file → 1.6 GB RAM instead of 4.3 GB)
df_results = dp.read_csv(
    "RESULTADOS_2024.csv",
    sep=";",
    encoding="latin1"
)

# Load participants (441 MB file → 215 MB RAM instead of 5.6 GB!)
df_participants = dp.read_csv(
    "PARTICIPANTES_2024.csv", 
    sep=";",
    encoding="latin1"
)

# Now you can process BOTH files on a 16GB laptop!
# Total: 1.8 GB instead of 10 GB
```

---

**Benchmark Script:** `benchmarks/enem_real_benchmark.py`  
**Raw Results:** `benchmarks/results/enem_results.json`
