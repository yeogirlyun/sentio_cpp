# Bug Report: Data Packing Issue in data_downloader.py

## Issue Summary
**Title**: Binary Data Packing Error - "pack expected 5 items for packing (got 6)"  
**Severity**: HIGH  
**Priority**: MEDIUM  
**Status**: OPEN  
**Affected Components**: Data Downloader, Binary File Generation, C++ Data Loading  

## Description
The `data_downloader.py` script successfully downloads and filters market data from Polygon.io, but fails when attempting to save data to binary format. The error occurs during the `struct.pack()` operation in the `save_to_bin()` function, where the script expects 5 items but receives 6 items for packing.

## Error Details

### Error Message
```
Error saving binary file: pack expected 5 items for packing (got 6)
```

### Location
- **File**: `tools/data_downloader.py`
- **Function**: `save_to_bin()`
- **Line**: 129-136 (struct.pack operation)

### Error Context
The error occurs when processing the following symbols:
- QQQ: 292,776 bars → Binary packing fails
- PSQ: 282,261 bars → Binary packing fails  
- TQQQ: 292,777 bars → Binary packing fails
- SQQQ: 292,776 bars → Binary packing fails

## Root Cause Analysis

### Code Analysis
The issue is in the `save_to_bin()` function at lines 129-136:

```python
# Define the struct format for the fixed-size data part of each bar
# q: int64_t, d: double, Q: uint64_t
bar_struct = struct.Struct('<qdddQ')  # 5 format specifiers

for row in df.itertuples():
    # ... string handling ...
    
    # Pack and write the fixed-size data
    packed_data = bar_struct.pack(
        row.ts_nyt_epoch,    # 1 item
        row.open,            # 2 items
        row.high,            # 3 items
        row.low,             # 4 items
        row.close,           # 5 items
        int(row.volume)      # 6 items - THIS CAUSES THE ERROR
    )
```

### Problem Identified
1. **Struct Format Mismatch**: The `bar_struct` is defined with 5 format specifiers (`<qdddQ`)
2. **Packing Arguments**: The `bar_struct.pack()` call provides 6 arguments
3. **Missing Format Specifier**: The struct definition is missing one format specifier for the volume field

### Expected vs Actual
- **Expected**: 5 items (ts_nyt_epoch, open, high, low, close)
- **Actual**: 6 items (ts_nyt_epoch, open, high, low, close, volume)
- **Missing**: Volume field format specifier in struct definition

## Impact Assessment

### Immediate Impact
- **Binary Files Corrupted**: All generated `.bin` files are only 37 bytes (corrupted)
- **C++ Loading Failure**: C++ backtester cannot load binary data
- **Fallback to CSV**: System falls back to slower CSV loading

### Data Integrity
- **CSV Files**: ✅ Working correctly (292K+ bars per symbol)
- **Binary Files**: ❌ Corrupted (37 bytes each)
- **Data Quality**: ✅ RTH filtering and holiday exclusion working

### Performance Impact
- **Loading Speed**: CSV loading is significantly slower than binary
- **Memory Usage**: Higher memory usage with CSV parsing
- **I/O Overhead**: Increased disk I/O with larger CSV files

## Technical Details

### Binary Format Specification
The intended binary format should be:
```
- uint64_t: Number of bars
- For each bar:
  - uint32_t: Length of ts_utc string
  - char[]: ts_utc string data
  - int64_t: ts_nyt_epoch
  - double: open, high, low, close
  - uint64_t: volume
```

### Current Struct Definition
```python
bar_struct = struct.Struct('<qdddQ')  # Missing volume field
# q: int64_t (ts_nyt_epoch)
# d: double (open)
# d: double (high) 
# d: double (low)
# d: double (close)
# Q: uint64_t (volume) - MISSING!
```

### Correct Struct Definition
```python
bar_struct = struct.Struct('<qddddQ')  # Added volume field
# q: int64_t (ts_nyt_epoch)
# d: double (open)
# d: double (high)
# d: double (low) 
# d: double (close)
# d: double (volume) - CORRECTED!
```

## Proposed Solutions

### Immediate Fix
1. **Update Struct Format**: Change `'<qdddQ'` to `'<qddddQ'` in line 120
2. **Test Binary Generation**: Verify binary files are created correctly
3. **Validate C++ Loading**: Ensure C++ backtester can load binary files

### Code Changes Required
```python
# Line 120: Fix struct format
bar_struct = struct.Struct('<qddddQ')  # Added 'd' for volume

# Line 135: Ensure volume is double, not int
packed_data = bar_struct.pack(
    row.ts_nyt_epoch,
    row.open,
    row.high,
    row.low,
    row.close,
    float(row.volume)  # Ensure volume is float/double
)
```

### Validation Steps
1. **Binary File Size**: Verify files are ~2-3MB (not 37 bytes)
2. **C++ Loading**: Test binary loading in C++ backtester
3. **Data Integrity**: Compare binary vs CSV data consistency

## Workaround
Currently using CSV files as a workaround:
- **Data Quality**: ✅ RTH filtering working correctly
- **Strategy Testing**: ✅ VWAPReversion working with 42K+ fills
- **Performance**: ⚠️ Slower than binary loading

## Files Affected

### Source Files
- `tools/data_downloader.py` - Main script with packing bug
- `src/polygon_client.cpp` - C++ binary loading logic
- `include/sentio/polygon_client.hpp` - C++ binary loading interface

### Generated Files
- `data/equities/*_RTH_NH.csv` - ✅ Working correctly
- `data/equities/*_RTH_NH.bin` - ❌ Corrupted (37 bytes each)

## Testing Results

### CSV Loading (Working)
```
VWAPReversion Results:
- Data verification passed
- 50,406 signals emitted
- 42,293 fills generated
- Performance: -6.81% return, -1.14 Sharpe ratio
```

### Binary Loading (Failed)
```
Error: Binary files corrupted (37 bytes each)
Fallback: Using CSV loading
```

## Priority
**MEDIUM** - CSV workaround is functional, but binary loading would improve performance significantly.

## Status
**OPEN** - Fix required for optimal performance.

## Assigned To
Development Team

## Created
2024-12-19

## Last Updated
2024-12-19

## Related Issues
- Zero orders/fills problem (IN PROGRESS)
- Strategy signal generation issues (IN PROGRESS)
- Data integrity verification (RESOLVED)

## Notes
- CSV files are working correctly and provide full functionality
- Binary files would provide significant performance improvement
- Fix is straightforward (single line change in struct format)
- No data loss or corruption in CSV files
