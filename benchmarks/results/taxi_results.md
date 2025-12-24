# NYC Taxi Data Benchmark Results

**Date:** 2025-12-24 02:39:08

**Dataset:** NYC Yellow Taxi Trip Records

## Individual Monthly Files

| File | Rows | File Size | Standard | Optimized | Saved | Reduction |
|------|------|-----------|----------|-----------|-------|----------|
| 2015-01 (January 2015) | 12,748,986 | 1.85 GB | 3,818 MB | 1,199 MB | 2,618 MB | **68.6%** |
| 2016-01 (January 2016) | 10,906,858 | 1.59 GB | 3,266 MB | 1,081 MB | 2,185 MB | **66.9%** |
| 2016-02 (February 2016) | 11,382,049 | 1.66 GB | 3,408 MB | 1,100 MB | 2,308 MB | **67.7%** |
| 2016-03 (March 2016) | 12,210,952 | 1.78 GB | 3,657 MB | 1,171 MB | 2,486 MB | **68.0%** |

### Performance Details

#### 2015-01 (January 2015)

- **Rows:** 12,748,986
- **Columns:** 19
- **File size:** 1.85 GB

**Memory Usage:**
- Standard: 3,817.73 MB
- Optimized: 1,199.32 MB
- **Saved: 2,618.42 MB (68.6% reduction)**

**Load Time:**
- Standard: 11.28s
- Optimized: 40.66s
- Overhead: 260.4%

**Top Column Optimizations:**

- `store_and_fwd_flag`: 583.60 MB saved (96.0% reduction)
- `tpep_pickup_datetime`: 555.51 MB saved (67.2% reduction)
- `tpep_dropoff_datetime`: 555.26 MB saved (67.2% reduction)
- `VendorID`: 85.11 MB saved (87.5% reduction)
- `passenger_count`: 85.11 MB saved (87.5% reduction)
- `RateCodeID`: 85.11 MB saved (87.5% reduction)
- `payment_type`: 85.11 MB saved (87.5% reduction)
- `trip_distance`: 48.63 MB saved (50.0% reduction)
- `pickup_longitude`: 48.63 MB saved (50.0% reduction)
- `pickup_latitude`: 48.63 MB saved (50.0% reduction)

#### 2016-01 (January 2016)

- **Rows:** 10,906,858
- **Columns:** 19
- **File size:** 1.59 GB

**Memory Usage:**
- Standard: 3,266.10 MB
- Optimized: 1,081.36 MB
- **Saved: 2,184.74 MB (66.9% reduction)**

**Load Time:**
- Standard: 8.14s
- Optimized: 29.59s
- Overhead: 263.8%

**Top Column Optimizations:**

- `store_and_fwd_flag`: 499.28 MB saved (96.0% reduction)
- `tpep_pickup_datetime`: 447.60 MB saved (63.3% reduction)
- `tpep_dropoff_datetime`: 447.34 MB saved (63.2% reduction)
- `VendorID`: 72.81 MB saved (87.5% reduction)
- `passenger_count`: 72.81 MB saved (87.5% reduction)
- `RatecodeID`: 72.81 MB saved (87.5% reduction)
- `payment_type`: 72.81 MB saved (87.5% reduction)
- `trip_distance`: 41.61 MB saved (50.0% reduction)
- `pickup_longitude`: 41.61 MB saved (50.0% reduction)
- `pickup_latitude`: 41.61 MB saved (50.0% reduction)

#### 2016-02 (February 2016)

- **Rows:** 11,382,049
- **Columns:** 19
- **File size:** 1.66 GB

**Memory Usage:**
- Standard: 3,408.40 MB
- Optimized: 1,100.45 MB
- **Saved: 2,307.95 MB (67.7% reduction)**

**Load Time:**
- Standard: 8.36s
- Optimized: 30.46s
- Overhead: 264.2%

**Top Column Optimizations:**

- `store_and_fwd_flag`: 521.03 MB saved (96.0% reduction)
- `tpep_pickup_datetime`: 481.08 MB saved (65.2% reduction)
- `tpep_dropoff_datetime`: 480.87 MB saved (65.2% reduction)
- `VendorID`: 75.98 MB saved (87.5% reduction)
- `passenger_count`: 75.98 MB saved (87.5% reduction)
- `RatecodeID`: 75.98 MB saved (87.5% reduction)
- `payment_type`: 75.98 MB saved (87.5% reduction)
- `trip_distance`: 43.42 MB saved (50.0% reduction)
- `pickup_longitude`: 43.42 MB saved (50.0% reduction)
- `pickup_latitude`: 43.42 MB saved (50.0% reduction)

#### 2016-03 (March 2016)

- **Rows:** 12,210,952
- **Columns:** 19
- **File size:** 1.78 GB

**Memory Usage:**
- Standard: 3,656.62 MB
- Optimized: 1,170.65 MB
- **Saved: 2,485.97 MB (68.0% reduction)**

**Load Time:**
- Standard: 8.98s
- Optimized: 33.14s
- Overhead: 269.2%

**Top Column Optimizations:**

- `store_and_fwd_flag`: 558.97 MB saved (96.0% reduction)
- `tpep_pickup_datetime`: 521.10 MB saved (65.8% reduction)
- `tpep_dropoff_datetime`: 520.86 MB saved (65.8% reduction)
- `VendorID`: 81.52 MB saved (87.5% reduction)
- `passenger_count`: 81.52 MB saved (87.5% reduction)
- `RatecodeID`: 81.52 MB saved (87.5% reduction)
- `payment_type`: 81.52 MB saved (87.5% reduction)
- `trip_distance`: 46.58 MB saved (50.0% reduction)
- `pickup_longitude`: 46.58 MB saved (50.0% reduction)
- `pickup_latitude`: 46.58 MB saved (50.0% reduction)

## Combined Multi-Month Dataset

**File:** Combined Dataset (2015-01 + 2016 Q1)

- **Total rows:** 47,248,845
- **Columns:** 19
- **File size:** 6.92 GB

### Memory Usage

- **Standard pandas:** 14,149 MB
- **diet-pandas:** 10,138 MB
- **Saved:** 4,010 MB (28.3% reduction)

### Load Time

- **Standard:** 54.10s
- **Optimized:** 111.31s
- **Overhead:** 105.8%

### Top Column Optimizations

| Column | Standard | Optimized | Saved | Reduction |
|--------|----------|-----------|-------|----------|
| `store_and_fwd_flag` | 2253.00 MB | 90.12 MB | 2162.88 MB | 96.0% |
| `passenger_count` | 360.48 MB | 45.06 MB | 315.42 MB | 87.5% |
| `RateCodeID` | 360.48 MB | 45.06 MB | 315.42 MB | 87.5% |
| `payment_type` | 360.48 MB | 45.06 MB | 315.42 MB | 87.5% |
| `trip_distance` | 360.48 MB | 180.24 MB | 180.24 MB | 50.0% |
| `pickup_longitude` | 360.48 MB | 180.24 MB | 180.24 MB | 50.0% |
| `pickup_latitude` | 360.48 MB | 180.24 MB | 180.24 MB | 50.0% |
| `dropoff_longitude` | 360.48 MB | 180.24 MB | 180.24 MB | 50.0% |
| `dropoff_latitude` | 360.48 MB | 180.24 MB | 180.24 MB | 50.0% |
| `fare_amount` | 360.48 MB | 180.24 MB | 180.24 MB | 50.0% |
| `extra` | 360.48 MB | 180.24 MB | 180.24 MB | 50.0% |
| `mta_tax` | 360.48 MB | 180.24 MB | 180.24 MB | 50.0% |
| `tip_amount` | 360.48 MB | 180.24 MB | 180.24 MB | 50.0% |
| `tolls_amount` | 360.48 MB | 180.24 MB | 180.24 MB | 50.0% |
| `improvement_surcharge` | 360.48 MB | 180.24 MB | 180.24 MB | 50.0% |

---

*Benchmark generated by diet-pandas*
