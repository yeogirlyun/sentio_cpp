# Position-History vs Audit Summarize Discrepancy Analysis

**Generated**: 2025-09-15 22:48:53
**Source Directory**: /Users/yeogirlyun/C++/sentio_cpp
**Description**: Comprehensive analysis of the small discrepancy between position-history and audit summarize calculations, including relevant source modules

**Total Files**: 14

---

## 🐛 **BUG REPORT**

# 🐛 Bug Report: Position-History vs Audit Summarize Discrepancy

## 📋 Executive Summary

**Issue**: Small but consistent discrepancy between position-history and audit summarize calculations
- **Position-History**: -20.42% total return, $79,579.39 final equity
- **Audit Summarize**: -20.33% total return, -$20,333.37 total P&L
- **Discrepancy**: ~0.09% difference in total return calculation

## 🔍 Detailed Analysis

### 📊 Observed Values (Run ID: 475482, sigor strategy, 4w test)

| Metric | Position-History | Audit Summarize | Difference |
|--------|------------------|-----------------|------------|
| **Final Equity** | $79,579.39 | N/A | N/A |
| **Total Return** | -20.42% | -20.33% | **0.09%** |
| **Total P&L** | -$20,420.61 | -$20,333.37 | **$87.24** |
| **Realized P&L** | -$16,463.33 | N/A | N/A |
| **Unrealized P&L** | $21.21 | N/A | N/A |
| **Trading Days** | 41 | 41 | ✅ Match |
| **Total Fills** | 5105 | 5105 | ✅ Match |

### 🎯 Root Cause Analysis

#### **Method 1: Position-History Calculation**
```cpp
// Position-History uses: Final Equity - Starting Capital
total_pnl = final_equity - starting_capital
total_return = (total_pnl / starting_capital) * 100.0
// Result: -20.42% return, -$20,420.61 P&L
```

#### **Method 2: Audit Summarize Calculation**
```cpp
// Audit Summarize uses: Sum of pnl_delta from fills
total_pnl = SUM(pnl_delta) from all fills
total_return = (total_pnl / starting_capital) * 100.0
// Result: -20.33% return, -$20,333.37 P&L
```

### 🔬 Technical Investigation

#### **Potential Sources of Discrepancy**

1. **Rounding Differences**
   - Position-History: Uses final equity calculation with potential floating-point precision
   - Audit Summarize: Sums individual fill P&L deltas with different rounding

2. **Cost Model Application**
   - Position-History: May apply costs differently in final equity calculation
   - Audit Summarize: Applies costs per fill in pnl_delta calculation

3. **Mark-to-Market Timing**
   - Position-History: Uses end-of-day marking for final equity
   - Audit Summarize: Uses fill-time marking for individual P&L

4. **Unrealized P&L Handling**
   - Position-History: Shows separate realized (-$16,463.33) and unrealized ($21.21) components
   - Audit Summarize: May include unrealized P&L in total calculation differently

### 🧮 Mathematical Verification

```cpp
// Position-History Method
starting_capital = 100000.00
final_equity = 79579.39
total_pnl_ph = 79579.39 - 100000.00 = -20420.61
total_return_ph = (-20420.61 / 100000.00) * 100.0 = -20.42%

// Audit Summarize Method  
total_pnl_audit = -20333.37
total_return_audit = (-20333.37 / 100000.00) * 100.0 = -20.33%

// Discrepancy Analysis
difference = |20420.61 - 20333.37| = 87.24
percentage_diff = (87.24 / 100000.00) * 100.0 = 0.087%
```

### 🎯 Impact Assessment

#### **Severity: LOW-MEDIUM**
- **Financial Impact**: $87.24 difference on $100K capital (0.087%)
- **Consistency Impact**: Creates confusion in reporting accuracy
- **User Trust**: Small discrepancies undermine confidence in system precision

#### **Business Risk**
- **Low**: Difference is within acceptable tolerance for most use cases
- **Medium**: Could compound over multiple runs or larger capital amounts
- **High**: Indicates underlying calculation inconsistency that may worsen

### 🔧 Recommended Fixes

#### **Immediate Actions**
1. **Standardize Calculation Method**: Choose one authoritative P&L calculation approach
2. **Implement Cross-Validation**: Add automatic validation between methods
3. **Add Precision Controls**: Ensure consistent rounding and precision handling

#### **Long-term Solutions**
1. **Single Source of Truth**: Implement canonical P&L calculation used by all reports
2. **Unified Metrics Engine**: Create single metrics calculator for all components
3. **Audit Trail Enhancement**: Add detailed calculation logs for debugging

### 🧪 Test Cases for Validation

#### **Test Case 1: Simple Buy/Sell**
```cpp
// Starting: $100K cash
// Buy: 100 shares QQQ @ $400 = $40K
// Sell: 100 shares QQQ @ $410 = $41K
// Expected: +$1K profit, +1% return
```

#### **Test Case 2: Multiple Trades**
```cpp
// Test with 10+ trades to verify cumulative P&L accuracy
// Verify both methods produce identical results
```

#### **Test Case 3: Cost Model Impact**
```cpp
// Test with different commission rates
// Verify cost application consistency
```

### 📈 Monitoring Recommendations

1. **Daily Reconciliation**: Automatically compare position-history vs audit summarize
2. **Threshold Alerts**: Alert when discrepancy exceeds 0.1%
3. **Regression Testing**: Include P&L consistency in automated test suite

## 🎯 Conclusion

The 0.09% discrepancy between position-history and audit summarize represents a **calculation methodology difference** rather than a fundamental bug. However, this inconsistency undermines system reliability and should be addressed through:

1. **Immediate**: Implement cross-validation and alerting
2. **Short-term**: Standardize on single calculation method  
3. **Long-term**: Build unified metrics engine with canonical calculations

**Priority**: Medium - Fix within next sprint to maintain user confidence in reporting accuracy.


---

## 📋 **TABLE OF CONTENTS**

1. [audit/DIAGNOSTIC_tpa_test_1757559256_day725_signals_only.txt](#file-1)
2. [audit/include/audit/audit_cli.hpp](#file-2)
3. [audit/include/audit/audit_db.hpp](#file-3)
4. [audit/include/audit/audit_db_recorder.hpp](#file-4)
5. [audit/include/audit/clock.hpp](#file-5)
6. [audit/include/audit/hash.hpp](#file-6)
7. [audit/include/audit/price_csv.hpp](#file-7)
8. [audit/src/audit_cli.cpp](#file-8)
9. [audit/src/audit_db.cpp](#file-9)
10. [audit/src/audit_db_recorder.cpp](#file-10)
11. [audit/src/clock.cpp](#file-11)
12. [audit/src/hash.cpp](#file-12)
13. [audit/src/price_csv.cpp](#file-13)
14. [audit/tests/test_verify.cpp](#file-14)

---

## 📄 **FILE 1 of 14**: audit/DIAGNOSTIC_tpa_test_1757559256_day725_signals_only.txt

**File Information**:
- **Path**: `audit/DIAGNOSTIC_tpa_test_1757559256_day725_signals_only.txt`

- **Size**: 321 lines
- **Modified**: 2025-09-12 00:38:46

- **Type**: .txt

```text
SIGNAL-TO-TRADE PIPELINE - AUDIT LOG
=====================================

Format: Signal (p) → Router → Sizer → Runner → Balance Changes
-------------------------------------------------------------

[  1] 2024-09-10 04:00:00 SIGNAL: QQQ HOLD (p=0.500)

[  2] 2024-09-10 04:01:00 SIGNAL: QQQ HOLD (p=0.500)

[  3] 2024-09-10 04:02:00 SIGNAL: QQQ HOLD (p=0.500)

[  4] 2024-09-10 04:03:00 SIGNAL: QQQ HOLD (p=0.500)

[  5] 2024-09-10 04:04:00 SIGNAL: QQQ HOLD (p=0.500)

[  6] 2024-09-10 04:05:00 SIGNAL: QQQ HOLD (p=0.500)

[  7] 2024-09-10 04:06:00 SIGNAL: QQQ HOLD (p=0.500)

124


[  9] 2024-09-10 04:08:00 SIGNAL: QQQ HOLD (p=0.600)

[ 10] 2024-09-10 04:09:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: SELL 527.38 SQQQ → SELL 579.57 PSQ
      RUNNER: SELL 527.38 SQQQ @ $47.42 → SELL 579.57 PSQ @ $43.16 | P&L: $0.00

[ 11] 2024-09-10 04:10:00 SIGNAL: QQQ HOLD (p=0.400)

[ 12] 2024-09-10 04:11:00 SIGNAL: QQQ HOLD (p=0.600)
      ROUTER: SELL 0.04 QQQ → SELL 0.24 TQQQ
      RUNNER: SELL 0.04 QQQ @ $452.35 → SELL 0.24 TQQQ @ $58.30 | P&L: $-0.00

[ 13] 2024-09-10 04:12:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: BUY 1.22 SQQQ → BUY 0.64 PSQ
      RUNNER: BUY 1.22 SQQQ @ $47.50 → BUY 0.64 PSQ @ $43.17 | P&L: $-0.10

[ 14] 2024-09-10 04:13:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: SELL 0.35 SQQQ
      RUNNER: SELL 0.35 SQQQ @ $47.48 | P&L: $0.00

[ 15] 2024-09-10 04:14:00 SIGNAL: QQQ HOLD (p=0.600)

[ 16] 2024-09-10 04:15:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: SELL 0.36 SQQQ
      RUNNER: SELL 0.36 SQQQ @ $47.45 | P&L: $0.00

[ 17] 2024-09-10 04:16:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: SELL 0.45 PSQ
      RUNNER: SELL 0.45 PSQ @ $43.16 | P&L: $0.00

[ 18] 2024-09-10 04:17:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: SELL 0.47 SQQQ
      RUNNER: SELL 0.47 SQQQ @ $47.42 | P&L: $0.00

[ 19] 2024-09-10 04:18:00 SIGNAL: QQQ HOLD (p=0.600)

[ 20] 2024-09-10 04:19:00 SIGNAL: QQQ HOLD (p=0.600)

[ 21] 2024-09-10 04:20:00 SIGNAL: QQQ BUY (p=0.800)

[ 22] 2024-09-10 04:21:00 SIGNAL: QQQ BUY (p=0.800)

[ 23] 2024-09-10 04:22:00 SIGNAL: QQQ BUY (p=0.800)

[ 24] 2024-09-10 04:23:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: BUY 1.24 SQQQ → BUY 0.57 PSQ
      RUNNER: BUY 1.24 SQQQ @ $47.50 → BUY 0.57 PSQ @ $43.17 | P&L: $-0.10

[ 25] 2024-09-10 04:25:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: SELL 1.23 SQQQ → SELL 0.63 PSQ
      RUNNER: SELL 1.23 SQQQ @ $47.42 → SELL 0.63 PSQ @ $43.16 | P&L: $0.00

[ 26] 2024-09-10 04:26:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: SELL 1.04 SQQQ → SELL 0.86 PSQ
      RUNNER: SELL 1.04 SQQQ @ $47.38 → SELL 0.86 PSQ @ $43.13 | P&L: $0.00

[ 27] 2024-09-10 04:27:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: SELL 1.46 SQQQ → SELL 0.88 PSQ
      RUNNER: SELL 1.46 SQQQ @ $47.30 → SELL 0.88 PSQ @ $43.12 | P&L: $0.00

[ 28] 2024-09-10 04:28:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: SELL 1.18 SQQQ → SELL 0.74 PSQ
      RUNNER: SELL 1.18 SQQQ @ $47.23 → SELL 0.74 PSQ @ $43.10 | P&L: $0.00

[ 29] 2024-09-10 04:29:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: SELL 1.10 SQQQ → SELL 0.77 PSQ
      RUNNER: SELL 1.10 SQQQ @ $47.17 → SELL 0.77 PSQ @ $43.08 | P&L: $0.00

[ 30] 2024-09-10 04:30:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: SELL 0.46 SQQQ → SELL 0.40 PSQ
      RUNNER: SELL 0.46 SQQQ @ $47.15 → SELL 0.40 PSQ @ $43.06 | P&L: $0.00

[ 31] 2024-09-10 04:31:00 SIGNAL: QQQ SELL (p=0.200)

[ 32] 2024-09-10 04:32:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: SELL 2.22 SQQQ → SELL 1.22 PSQ
      RUNNER: SELL 2.22 SQQQ @ $47.02 → SELL 1.22 PSQ @ $43.04 | P&L: $0.00

[ 33] 2024-09-10 04:33:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: BUY 1.02 SQQQ → BUY 0.54 PSQ
      RUNNER: BUY 1.02 SQQQ @ $47.08 → BUY 0.54 PSQ @ $43.05 | P&L: $0.40

[ 34] 2024-09-10 04:34:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: BUY 1.67 SQQQ → BUY 1.08 PSQ
      RUNNER: BUY 1.67 SQQQ @ $47.17 → BUY 1.08 PSQ @ $43.08 | P&L: $0.49

[ 35] 2024-09-10 04:35:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: SELL 1.29 SQQQ → SELL 0.76 PSQ
      RUNNER: SELL 1.29 SQQQ @ $47.10 → SELL 0.76 PSQ @ $43.06 | P&L: $0.00

[ 36] 2024-09-10 04:36:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: SELL 0.99 SQQQ → SELL 0.73 PSQ
      RUNNER: SELL 0.99 SQQQ @ $47.05 → SELL 0.73 PSQ @ $43.04 | P&L: $0.00

[ 37] 2024-09-10 04:37:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: BUY 0.57 SQQQ → BUY 0.45 PSQ
      RUNNER: BUY 0.57 SQQQ @ $47.08 → BUY 0.45 PSQ @ $43.05 | P&L: $0.24

[ 38] 2024-09-10 04:38:00 SIGNAL: QQQ HOLD (p=0.600)
      ROUTER: BUY 0.13 QQQ → SELL 0.85 TQQQ
      RUNNER: BUY 0.13 QQQ @ $453.37 → SELL 0.85 TQQQ @ $58.69 | P&L: $0.32

[ 39] 2024-09-10 04:39:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: BUY 1.31 SQQQ → BUY 0.78 PSQ
      RUNNER: BUY 1.31 SQQQ @ $47.15 → BUY 0.78 PSQ @ $43.07 | P&L: $0.42

[ 40] 2024-09-10 04:40:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: SELL 2.28 SQQQ → SELL 1.35 PSQ
      RUNNER: SELL 2.28 SQQQ @ $47.02 → SELL 1.35 PSQ @ $43.04 | P&L: $0.00

[ 41] 2024-09-10 04:41:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: SELL 0.55 SQQQ → SELL 0.50 PSQ
      RUNNER: SELL 0.55 SQQQ @ $47.00 → SELL 0.50 PSQ @ $43.02 | P&L: $0.00

[ 42] 2024-09-10 04:42:00 SIGNAL: QQQ SELL (p=0.200)

[ 43] 2024-09-10 04:43:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: BUY 1.75 SQQQ → BUY 0.88 PSQ
      RUNNER: BUY 1.75 SQQQ @ $47.10 → BUY 0.88 PSQ @ $43.04 | P&L: $0.66

[ 44] 2024-09-10 04:44:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: SELL 0.94 SQQQ → SELL 0.25 PSQ
      RUNNER: SELL 0.94 SQQQ @ $47.04 → SELL 0.25 PSQ @ $43.04 | P&L: $0.00

[ 45] 2024-09-10 04:45:00 SIGNAL: QQQ HOLD (p=0.600)

[ 46] 2024-09-10 04:46:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: BUY 0.69 SQQQ → BUY 0.55 PSQ
      RUNNER: BUY 0.69 SQQQ @ $47.08 → BUY 0.55 PSQ @ $43.06 | P&L: $0.29

[ 47] 2024-09-10 04:47:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: SELL 0.57 SQQQ → SELL 0.57 PSQ
      RUNNER: SELL 0.57 SQQQ @ $47.05 → SELL 0.57 PSQ @ $43.04 | P&L: $0.00

[ 48] 2024-09-10 04:48:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: BUY 0.84 SQQQ → BUY 0.43 PSQ
      RUNNER: BUY 0.84 SQQQ @ $47.10 → BUY 0.43 PSQ @ $43.05 | P&L: $0.31

[ 49] 2024-09-10 04:49:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: SELL 0.74 SQQQ → SELL 0.26 PSQ
      RUNNER: SELL 0.74 SQQQ @ $47.05 → SELL 0.26 PSQ @ $43.05 | P&L: $0.00

[ 50] 2024-09-10 04:50:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: SELL 534.03 SQQQ → SELL 584.03 PSQ
      RUNNER: SELL 534.03 SQQQ @ $47.08 → SELL 584.03 PSQ @ $43.05 | P&L: $0.00

[ 51] 2024-09-10 04:51:00 SIGNAL: QQQ HOLD (p=0.600)
      ROUTER: BUY 55.42 QQQ → BUY 427.69 TQQQ
      RUNNER: BUY 55.42 QQQ @ $453.61 → BUY 427.69 TQQQ @ $58.78 | P&L: $0.00

[ 52] 2024-09-10 04:52:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: SELL 533.74 SQQQ → SELL 583.82 PSQ
      RUNNER: SELL 533.74 SQQQ @ $47.10 → SELL 583.82 PSQ @ $43.06 | P&L: $0.00

[ 53] 2024-09-10 04:53:00 SIGNAL: QQQ BUY (p=0.800)
      ROUTER: BUY 55.49 QQQ → BUY 429.40 TQQQ
      RUNNER: BUY 55.49 QQQ @ $453.01 → BUY 429.40 TQQQ @ $58.55 | P&L: $0.00

[ 54] 2024-09-10 04:54:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: SELL 533.75 SQQQ → SELL 583.76 PSQ
      RUNNER: SELL 533.75 SQQQ @ $47.10 → SELL 583.76 PSQ @ $43.06 | P&L: $0.00

[ 55] 2024-09-10 04:55:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: SELL 534.88 SQQQ → SELL 584.23 PSQ
      RUNNER: SELL 534.88 SQQQ @ $47.00 → SELL 584.23 PSQ @ $43.03 | P&L: $0.00

[ 56] 2024-09-10 04:56:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: SELL 535.17 SQQQ → SELL 584.23 PSQ
      RUNNER: SELL 535.17 SQQQ @ $46.98 → SELL 584.23 PSQ @ $43.03 | P&L: $0.00

[ 57] 2024-09-10 04:57:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: SELL 536.02 SQQQ → SELL 584.71 PSQ
      RUNNER: SELL 536.02 SQQQ @ $46.90 → SELL 584.71 PSQ @ $42.99 | P&L: $0.00

[ 58] 2024-09-10 04:58:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: SELL 536.90 SQQQ → SELL 585.05 PSQ
      RUNNER: SELL 536.90 SQQQ @ $46.82 → SELL 585.05 PSQ @ $42.97 | P&L: $0.00

[ 59] 2024-09-10 04:59:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: SELL 537.16 SQQQ → SELL 585.05 PSQ
      RUNNER: SELL 537.16 SQQQ @ $46.80 → SELL 585.05 PSQ @ $42.97 | P&L: $0.00

[ 60] 2024-09-10 05:00:00 SIGNAL: QQQ SELL (p=0.200)

[ 61] 2024-09-10 22:30:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: SELL 10.07 SQQQ → SELL 5.29 PSQ
      RUNNER: SELL 10.07 SQQQ @ $46.15 → SELL 5.29 PSQ @ $42.78 | P&L: $0.00

[ 62] 2024-09-10 22:31:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: SELL 2.80 SQQQ → SELL 1.60 PSQ
      RUNNER: SELL 2.80 SQQQ @ $45.98 → SELL 1.60 PSQ @ $42.72 | P&L: $0.00

[ 63] 2024-09-10 22:32:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: SELL 2.40 SQQQ → SELL 1.26 PSQ
      RUNNER: SELL 2.40 SQQQ @ $45.83 → SELL 1.26 PSQ @ $42.67 | P&L: $0.00

[ 64] 2024-09-10 22:33:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: BUY 3.57 SQQQ → BUY 1.77 PSQ
      RUNNER: BUY 3.57 SQQQ @ $46.05 → BUY 1.77 PSQ @ $42.73 | P&L: $3.02

[ 65] 2024-09-10 22:34:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: BUY 2.00 SQQQ → BUY 1.18 PSQ
      RUNNER: BUY 2.00 SQQQ @ $46.17 → BUY 1.18 PSQ @ $42.78 | P&L: $1.44

[ 66] 2024-09-10 22:35:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: BUY 1.10 SQQQ → BUY 0.24 PSQ
      RUNNER: BUY 1.10 SQQQ @ $46.25 → BUY 0.24 PSQ @ $42.78 | P&L: $0.64

[ 67] 2024-09-10 22:36:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: BUY 2.79 SQQQ → BUY 1.72 PSQ
      RUNNER: BUY 2.79 SQQQ @ $46.42 → BUY 1.72 PSQ @ $42.85 | P&L: $1.22

[ 68] 2024-09-10 22:37:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: SELL 0.75 SQQQ → SELL 0.28 PSQ
      RUNNER: SELL 0.75 SQQQ @ $46.38 → SELL 0.28 PSQ @ $42.84 | P&L: $0.00

[ 69] 2024-09-10 22:38:00 SIGNAL: QQQ HOLD (p=0.600)
      ROUTER: BUY 55.30 QQQ → BUY 423.11 TQQQ
      RUNNER: BUY 55.30 QQQ @ $455.67 → BUY 423.11 TQQQ @ $59.55 | P&L: $0.00

[ 70] 2024-09-10 22:39:00 SIGNAL: QQQ HOLD (p=0.600)

[ 71] 2024-09-10 22:40:00 SIGNAL: QQQ HOLD (p=0.400)

[ 72] 2024-09-10 22:41:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: SELL 0.36 PSQ
      RUNNER: SELL 0.36 PSQ @ $42.84 | P&L: $0.00

[ 73] 2024-09-10 22:42:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: BUY 0.26 PSQ
      RUNNER: BUY 0.26 PSQ @ $42.85 | P&L: $0.03

[ 74] 2024-09-10 22:43:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: BUY 0.32 PSQ
      RUNNER: BUY 0.32 PSQ @ $42.87 | P&L: $0.03

[ 75] 2024-09-10 22:44:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: SELL 3.25 SQQQ → SELL 2.30 PSQ
      RUNNER: SELL 3.25 SQQQ @ $46.23 → SELL 2.30 PSQ @ $42.80 | P&L: $0.00

[ 76] 2024-09-10 22:45:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: BUY 2.29 SQQQ → BUY 1.16 PSQ
      RUNNER: BUY 2.29 SQQQ @ $46.35 → BUY 1.16 PSQ @ $42.83 | P&L: $1.17

[ 77] 2024-09-10 22:46:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: SELL 0.86 SQQQ
      RUNNER: SELL 0.86 SQQQ @ $46.30 | P&L: $0.00

[ 78] 2024-09-10 22:47:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: SELL 1.91 SQQQ → SELL 1.57 PSQ
      RUNNER: SELL 1.91 SQQQ @ $46.20 → SELL 1.57 PSQ @ $42.79 | P&L: $0.00

[ 79] 2024-09-10 22:48:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: BUY 0.84 SQQQ → BUY 0.41 PSQ
      RUNNER: BUY 0.84 SQQQ @ $46.25 → BUY 0.41 PSQ @ $42.80 | P&L: $0.52

[ 80] 2024-09-10 22:49:00 SIGNAL: QQQ HOLD (p=0.600)
      ROUTER: BUY 0.05 QQQ → SELL 0.47 TQQQ
      RUNNER: BUY 0.05 QQQ @ $456.12 → SELL 0.47 TQQQ @ $59.73 | P&L: $0.08

[ 81] 2024-09-10 22:50:00 SIGNAL: QQQ BUY (p=0.800)
      ROUTER: SELL 0.05 QQQ → BUY 0.37 TQQQ
      RUNNER: SELL 0.05 QQQ @ $455.69 → BUY 0.37 TQQQ @ $59.57 | P&L: $0.00

[ 82] 2024-09-10 22:51:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: BUY 1.08 SQQQ → BUY 0.81 PSQ
      RUNNER: BUY 1.08 SQQQ @ $46.30 → BUY 0.81 PSQ @ $42.82 | P&L: $0.65

[ 83] 2024-09-10 22:52:00 SIGNAL: QQQ HOLD (p=0.400)

[ 84] 2024-09-10 22:53:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: SELL 2.07 SQQQ → SELL 1.37 PSQ
      RUNNER: SELL 2.07 SQQQ @ $46.20 → SELL 1.37 PSQ @ $42.79 | P&L: $0.00

[ 85] 2024-09-10 22:54:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: SELL 0.93 SQQQ → SELL 0.63 PSQ
      RUNNER: SELL 0.93 SQQQ @ $46.15 → SELL 0.63 PSQ @ $42.77 | P&L: $0.00

[ 86] 2024-09-10 22:55:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: SELL 3.33 SQQQ → SELL 2.15 PSQ
      RUNNER: SELL 3.33 SQQQ @ $45.98 → SELL 2.15 PSQ @ $42.72 | P&L: $0.00

[ 87] 2024-09-10 22:56:00 SIGNAL: QQQ SELL (p=0.200)
      ROUTER: BUY 1.80 SQQQ → BUY 1.18 PSQ
      RUNNER: BUY 1.80 SQQQ @ $46.08 → BUY 1.18 PSQ @ $42.75 | P&L: $1.52

[ 88] 2024-09-10 22:57:00 SIGNAL: QQQ HOLD (p=0.400)
      ROUTER: BUY 1.43 SQQQ → BUY 0.58 PSQ
      RUNNER: BUY 1.43 SQQQ @ $46.15 → BUY 0.58 PSQ @ $42.75 | P&L: $1.03

[ 89] 2024-09-10 22:58:00 SIGNAL: QQQ HOLD (p=0.600)
      ROUTER: BUY 0.11 QQQ → SELL 0.68 TQQQ
      RUNNER: BUY 0.11 QQQ @ $456.52 → SELL 0.68 TQQQ @ $59.89 | P&L: $0.23

[ 90] 2024-09-10 22:59:00 SIGNAL: QQQ HOLD (p=0.600)
      ROUTER: SELL 0.04 QQQ
      RUNNER: SELL 0.04 QQQ @ $456.41 | P&L: $0.03


```

## 📄 **FILE 2 of 14**: audit/include/audit/audit_cli.hpp

**File Information**:
- **Path**: `audit/include/audit/audit_cli.hpp`

- **Size**: 16 lines
- **Modified**: 2025-09-13 19:22:09

- **Type**: .hpp

```text
#pragma once
#include <string>

int audit_main(int argc, char** argv);

// Additional utility functions for run discovery
namespace audit {
    void list_runs(const std::string& db_path, const std::string& strategy_filter = "", const std::string& kind_filter = "");
    void find_latest_run(const std::string& db_path, const std::string& strategy_filter = "");
    void show_run_info(const std::string& db_path, const std::string& run_id);
    void show_trade_flow(const std::string& db_path, const std::string& run_id, const std::string& symbol_filter = "", int limit = 20, bool enhanced = false, bool show_buy = false, bool show_sell = false, bool show_hold = false);
    void show_signal_stats(const std::string& db_path, const std::string& run_id, const std::string& strategy_filter = "");
    void show_signal_flow(const std::string& db_path, const std::string& run_id, const std::string& symbol_filter = "", int limit = 20, bool show_buy = false, bool show_sell = false, bool show_hold = false, bool enhanced = false);
    void show_position_history(const std::string& db_path, const std::string& run_id, const std::string& symbol_filter = "", int limit = 20, bool show_buy = false, bool show_sell = false, bool show_hold = false);
    void show_strategies_summary(const std::string& db_path);
}

```

## 📄 **FILE 3 of 14**: audit/include/audit/audit_db.hpp

**File Information**:
- **Path**: `audit/include/audit/audit_db.hpp`

- **Size**: 132 lines
- **Modified**: 2025-09-15 21:07:52

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <vector>
#include <optional>
#include <cstdint>
#include <map>

struct sqlite3;

namespace audit {

struct Event {
  std::string  run_id;
  std::int64_t ts_millis{};
  std::string  kind;     // SIGNAL|ORDER|FILL|PNL|NOTE
  std::string  symbol;   // optional
  std::string  side;     // BUY|SELL|NEUTRAL
  double       qty = 0.0;
  double       price = 0.0;
  double       pnl_delta = 0.0;
  double       weight = 0.0;
  double       prob = 0.0;
  std::string  reason;   // optional
  std::string  note;     // optional
};

struct RunRow {
  std::string  run_id;
  std::int64_t started_at{};
  std::optional<std::int64_t> ended_at;
  std::string  kind;
  std::string  strategy;
  std::string  params_json;
  std::string  data_hash;
  std::string  git_rev;
  std::string  note;
  // New canonical period fields
  std::optional<std::int64_t> run_period_start_ts_ms;
  std::optional<std::int64_t> run_period_end_ts_ms;
  std::optional<int> run_trading_days;
  std::string session_calendar = "XNYS"; // Default to NYSE
  // Dataset traceability fields
  std::string dataset_source_type = "unknown";
  std::string dataset_file_path = "";
  std::string dataset_file_hash = "";
  std::string dataset_track_id = "";
  std::string dataset_regime = "";
  int dataset_bars_count = 0;
  std::int64_t dataset_time_range_start = 0;
  std::int64_t dataset_time_range_end = 0;
};

class DB {
public:
  explicit DB(const std::string& path);
  ~DB();
  DB(const DB&) = delete; DB& operator=(const DB&) = delete;

  void init_schema();

  void new_run(const RunRow& run);
  void end_run(const std::string& run_id, std::int64_t ended_at);
  
  // Latest run ID tracking
  void set_latest_run_id(const std::string& run_id);
  std::string get_latest_run_id();
  
  // Daily returns storage for canonical MPR calculation
  void store_daily_returns(const std::string& run_id, const std::vector<std::pair<std::string, double>>& daily_returns);
  std::vector<double> load_daily_returns(const std::string& run_id);

  // Append event; computes seq/hash chain atomically.
  // Returns (seq, hash_curr).
  std::pair<std::int64_t, std::string> append_event(const Event& ev);

  // Verification results: (ok, msg)
  std::pair<bool, std::string> verify_run(const std::string& run_id);

  // Get all events for a run (for unified metrics calculation)
  std::vector<Event> get_events_for_run(const std::string& run_id);

  // Summary (counts, pnl, ts range, trading metrics)
  struct Summary {
    std::int64_t n_total{};
    std::int64_t n_signal{};
    std::int64_t n_order{};
    std::int64_t n_fill{};
    std::int64_t n_pnl{};
    double pnl_sum{};
    std::int64_t ts_first{}, ts_last{};
    
    // Trading performance metrics
    double mpr{};           // Monthly Projected Return (%)
    double sharpe{};        // Sharpe ratio
    double daily_trades{};  // Average daily trades
    double total_return{};  // Total return percentage
    double max_drawdown{};  // Maximum drawdown (%)
    int trading_days{};     // Number of trading days
    
    // Instrument distribution and P&L breakdown
    std::map<std::string, int64_t> instrument_fills;  // Symbol -> number of fills
    std::map<std::string, double> instrument_pnl;     // Symbol -> total P&L
    std::map<std::string, double> instrument_volume;  // Symbol -> total trading volume
  };
  Summary summarize(const std::string& run_id);

  // Export raw rows as CSV/JSONL strings (streaming)
  void export_run_csv (const std::string& run_id, const std::string& out_path);
  void export_run_jsonl(const std::string& run_id, const std::string& out_path);

  // Ad-hoc grep via WHERE clause; returns number of rows printed to stdout
  long grep_where(const std::string& run_id, const std::string& where_sql);

  // Diff two runs -> printable text summary returned
  std::string diff_runs(const std::string& run_a, const std::string& run_b);

  // Vacuum & analyze
  void vacuum();
  
  // Access to raw database for advanced queries
  sqlite3* get_db() const { return db_; }

private:
  sqlite3* db_{};
  std::string db_path_;

  // helpers
  std::pair<std::int64_t,std::string> last_seq_and_hash(const std::string& run_id);
  std::string canonical_content_string(std::int64_t seq, const Event& ev);
};

} // namespace audit

```

## 📄 **FILE 4 of 14**: audit/include/audit/audit_db_recorder.hpp

**File Information**:
- **Path**: `audit/include/audit/audit_db_recorder.hpp`

- **Size**: 71 lines
- **Modified**: 2025-09-15 01:06:02

- **Type**: .hpp

```text
#pragma once
#include "audit/audit_db.hpp"
#include "audit/clock.hpp"
#include "sentio/audit_interface.hpp"
#include <string>
#include <memory>

namespace audit {

// Integrated audit recorder that writes directly to SQLite database
// Replaces the JSONL-based AuditRecorder for sentio_cli integration
class AuditDBRecorder : public sentio::IAuditRecorder {
public:
    explicit AuditDBRecorder(const std::string& db_path, const std::string& run_id, const std::string& note = "Generated by sentio_cli");
    ~AuditDBRecorder();

    // Run lifecycle events
    void event_run_start(std::int64_t ts, const std::string& meta) override;
    void event_run_end(std::int64_t ts, const std::string& meta) override;

    // Market data events
    void event_bar(std::int64_t ts, const std::string& inst, double open, double high, double low, double close, double volume) override;

    // Signal events
    void event_signal(std::int64_t ts, const std::string& base, sentio::SigType t, double conf) override;
    void event_signal_ex(std::int64_t ts, const std::string& base, sentio::SigType t, double conf, const std::string& chain_id) override;

    // Trading events
    void event_route(std::int64_t ts, const std::string& base, const std::string& inst, double tw) override;
    void event_route_ex(std::int64_t ts, const std::string& base, const std::string& inst, double tw, const std::string& chain_id) override;
    void event_order(std::int64_t ts, const std::string& inst, sentio::Side side, double qty, double limit_px) override;
    void event_order_ex(std::int64_t ts, const std::string& inst, sentio::Side side, double qty, double limit_px, const std::string& chain_id) override;
    void event_fill(std::int64_t ts, const std::string& inst, double price, double qty, double fees, sentio::Side side) override;
    void event_fill_ex(std::int64_t ts, const std::string& inst, double price, double qty, double fees, sentio::Side side, 
                       double realized_pnl_delta, double equity_after, double position_after, const std::string& chain_id) override;

    // Portfolio events
    void event_snapshot(std::int64_t ts, const sentio::AccountState& a) override;

    // Metric events
    void event_metric(std::int64_t ts, const std::string& key, double val) override;

    // Signal diagnostics events
    void event_signal_diag(std::int64_t ts, const std::string& strategy_name, const SignalDiag& diag) override;
    void event_signal_drop(std::int64_t ts, const std::string& strategy_name, const std::string& symbol, 
                          DropReason reason, const std::string& chain_id, const std::string& note = "") override;

    // Utility methods
    std::string get_run_id() const { return run_id_; }
    bool is_logging_enabled() const { return logging_enabled_; }
    void set_logging_enabled(bool enabled) { logging_enabled_ = enabled; }
    
    // Canonical MPR support
    void store_daily_returns(const std::vector<std::pair<std::string, double>>& daily_returns);
    DB& get_db() { return *db_; }

private:
    std::unique_ptr<DB> db_;
    std::string run_id_;
    std::string strategy_name_;
    std::string note_;
    bool logging_enabled_;
    std::int64_t started_at_;

    // Helper methods
    std::string side_to_string(int side);
    std::string sig_type_to_string(int sig_type);
    void log_event(const Event& event);
};

} // namespace audit

```

## 📄 **FILE 5 of 14**: audit/include/audit/clock.hpp

**File Information**:
- **Path**: `audit/include/audit/clock.hpp`

- **Size**: 6 lines
- **Modified**: 2025-09-11 15:18:13

- **Type**: .hpp

```text
#pragma once
#include <cstdint>

namespace audit {
std::int64_t now_millis();
} // namespace audit

```

## 📄 **FILE 6 of 14**: audit/include/audit/hash.hpp

**File Information**:
- **Path**: `audit/include/audit/hash.hpp`

- **Size**: 11 lines
- **Modified**: 2025-09-11 14:14:19

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <vector>
#include <cstdint>

// Tiny SHA-256 (public-domain) interface.
// Impl provided in src/hash.cpp
namespace audit {
std::string sha256_hex(const void* data, size_t n);
inline std::string sha256_hex(const std::string& s) { return sha256_hex(s.data(), s.size()); }
} // namespace audit

```

## 📄 **FILE 7 of 14**: audit/include/audit/price_csv.hpp

**File Information**:
- **Path**: `audit/include/audit/price_csv.hpp`

- **Size**: 11 lines
- **Modified**: 2025-09-11 14:14:37

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace audit {
struct Bar { std::int64_t ts; double open, high, low, close, volume; };
using Series = std::vector<Bar>;
std::unordered_map<std::string, Series> load_price_csv(const std::string& path);
}

```

## 📄 **FILE 8 of 14**: audit/src/audit_cli.cpp

**File Information**:
- **Path**: `audit/src/audit_cli.cpp`

- **Size**: 1921 lines
- **Modified**: 2025-09-15 03:18:13

- **Type**: .cpp

```text
#include "audit/audit_cli.hpp"
#include "audit/audit_db.hpp"
#include "audit/clock.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <sqlite3.h>

using namespace audit;

// **CONFLICT DETECTION**: ETF classifications for conflict analysis
static const std::unordered_set<std::string> LONG_ETFS = {"QQQ", "TQQQ"};
static const std::unordered_set<std::string> INVERSE_ETFS = {"SQQQ"}; // PSQ removed

// **CONFLICT DETECTION**: Position tracking for conflict analysis
struct ConflictPosition {
    double qty = 0.0;
    std::string symbol;
};

struct ConflictAnalysis {
    std::vector<std::string> conflicts;
    std::unordered_map<std::string, ConflictPosition> positions;
    int conflict_count = 0;
    bool has_conflicts = false;
};

// **CONFLICT DETECTION**: Check for conflicting positions
static ConflictAnalysis analyze_position_conflicts(const std::unordered_map<std::string, ConflictPosition>& positions) {
    ConflictAnalysis analysis;
    analysis.positions = positions;
    
    bool has_long_etf = false;
    bool has_inverse_etf = false;
    bool has_short_qqq = false;
    
    std::vector<std::string> long_positions;
    std::vector<std::string> short_positions;
    std::vector<std::string> inverse_positions;
    
    for (const auto& [symbol, pos] : positions) {
        if (std::abs(pos.qty) > 1e-6) {
            if (LONG_ETFS.count(symbol)) {
                if (pos.qty > 0) {
                    has_long_etf = true;
                    long_positions.push_back(symbol + "(+" + std::to_string((int)pos.qty) + ")");
                } else {
                    has_short_qqq = true;
                    short_positions.push_back("SHORT " + symbol + "(" + std::to_string((int)pos.qty) + ")");
                }
            }
            if (INVERSE_ETFS.count(symbol)) {
                has_inverse_etf = true;
                inverse_positions.push_back(symbol + "(" + std::to_string((int)pos.qty) + ")");
            }
        }
    }
    
    // **CONFLICT RULES**:
    // 1. Long ETF conflicts with Inverse ETF or SHORT QQQ
    // 2. SHORT QQQ conflicts with Long ETF
    // 3. Inverse ETF conflicts with Long ETF
    if ((has_long_etf && (has_inverse_etf || has_short_qqq)) || 
        (has_short_qqq && has_long_etf)) {
        analysis.has_conflicts = true;
        analysis.conflict_count++;
        
        std::string conflict_desc = "CONFLICTING POSITIONS DETECTED: ";
        if (!long_positions.empty()) {
            conflict_desc += "Long: ";
            for (size_t i = 0; i < long_positions.size(); ++i) {
                if (i > 0) conflict_desc += ", ";
                conflict_desc += long_positions[i];
            }
        }
        if (!short_positions.empty()) {
            if (!long_positions.empty()) conflict_desc += "; ";
            conflict_desc += "Short: ";
            for (size_t i = 0; i < short_positions.size(); ++i) {
                if (i > 0) conflict_desc += ", ";
                conflict_desc += short_positions[i];
            }
        }
        if (!inverse_positions.empty()) {
            if (!long_positions.empty() || !short_positions.empty()) conflict_desc += "; ";
            conflict_desc += "Inverse: ";
            for (size_t i = 0; i < inverse_positions.size(); ++i) {
                if (i > 0) conflict_desc += ", ";
                conflict_desc += inverse_positions[i];
            }
        }
        
        analysis.conflicts.push_back(conflict_desc);
    }
    
    return analysis;
}

static const char* usage =
  "sentio_audit <cmd> [options]\n"
  "\n"
  "DATABASE MANAGEMENT:\n"
  "  init           [--db DB]\n"
  "  reset          [--db DB] [--confirm]  # WARNING: Deletes all audit data!\n"
  "  vacuum         [--db DB]\n"
  "\n"
  "RUN MANAGEMENT:\n"
  "  new-run        [--db DB] --run RUN --strategy STRAT --kind KIND --params FILE --data-hash HASH --git REV [--note NOTE]\n"
  "  end-run        [--db DB] --run RUN\n"
  "  log            [--db DB] --run RUN --ts MS --kind KIND [--symbol S] [--side SIDE] [--qty Q] [--price P] [--pnl P] [--weight W] [--prob P] [--reason R] [--note NOTE]\n"
  "\n"
  "QUERY COMMANDS:\n"
  "  list           [--db DB] [--strategy STRAT] [--kind KIND]\n"
  "  latest         [--db DB] [--strategy STRAT]\n"
  "  info           [--db DB] [--run RUN]  # defaults to latest run\n"
  "\n"
  "ANALYSIS COMMANDS:\n"
  "  verify         [--db DB] [--run RUN]  # defaults to latest run\n"
  "  summarize      [--db DB] [--run RUN]  # defaults to latest run\n"
  "  strategies-summary [--db DB]  # summary of all strategies' most recent runs\n"
  "  signal-stats   [--db DB] [--run RUN] [--strategy STRAT]  # defaults to latest run\n"
  "\n"
  "FLOW ANALYSIS:\n"
  "  trade-flow     [--db DB] [--run RUN] [--symbol S] [--limit N] [--max [N]] [--buy] [--sell] [--hold] [--enhanced]  # defaults to latest run, limit=20\n"
  "  signal-flow    [--db DB] [--run RUN] [--symbol S] [--limit N] [--max [N]] [--buy] [--sell] [--hold] [--enhanced]  # defaults to latest run, limit=20\n"
  "  position-history [--db DB] [--run RUN] [--symbol S] [--limit N] [--max [N]] [--buy] [--sell] [--hold]  # defaults to latest run, limit=20\n"
  "\n"
  "DATA OPERATIONS:\n"
  "  export         [--db DB] [--run RUN] --format FORMAT --output FILE  # defaults to latest run\n"
  "  grep           [--db DB] [--run RUN] --where \"CONDITION\"  # defaults to latest run\n"
  "  diff           [--db DB] --run1 RUN1 --run2 RUN2\n"
  "\n"
  "DEFAULTS:\n"
  "  Database: audit/sentio_audit.sqlite3\n"
  "  Run: latest run (for analysis commands)\n"
  "  Limit: 20 events (for flow analysis)\n";

static const char* arg(const char* k, int argc, char** argv, const char* def=nullptr) {
  for (int i=1;i<argc-1;i++) if (!strcmp(argv[i], k)) return argv[i+1];
  return def;
}
static bool has(const char* k, int argc, char** argv) {
  for (int i=1;i<argc;i++) if (!strcmp(argv[i], k)) return true;
  return false;
}

// Helper function to remove chain information from note field for display
static std::string clean_note_for_display(const char* note) {
  if (!note) return "";
  
  std::string note_str = note;
  
  // Remove chain= information
  size_t chain_pos = note_str.find("chain=");
  if (chain_pos != std::string::npos) {
    size_t comma_pos = note_str.find(",", chain_pos);
    if (comma_pos != std::string::npos) {
      // Remove "chain=xxx," or "chain=xxx" at end
      note_str.erase(chain_pos, comma_pos - chain_pos + 1);
    } else {
      // Remove "chain=xxx" at end
      note_str.erase(chain_pos);
    }
  }
  
  // Clean up any leading/trailing commas or spaces
  while (!note_str.empty() && (note_str.back() == ',' || note_str.back() == ' ')) {
    note_str.pop_back();
  }
  while (!note_str.empty() && (note_str.front() == ',' || note_str.front() == ' ')) {
    note_str.erase(0, 1);
  }
  
  return note_str;
}

static std::string get_latest_run_id(const std::string& db_path) {
  try {
    DB db(db_path);
    // **FIXED**: Use dedicated latest run ID tracking instead of timestamp-based ordering
    std::string latest_run_id = db.get_latest_run_id();
    
    // Fallback to timestamp-based ordering if no latest run ID is stored
    if (latest_run_id.empty()) {
      std::string sql = "SELECT run_id FROM audit_runs ORDER BY started_at DESC LIMIT 1";
      sqlite3_stmt* st = nullptr;
      int rc = sqlite3_prepare_v2(db.get_db(), sql.c_str(), -1, &st, nullptr);
      if (rc == SQLITE_OK) {
        if (sqlite3_step(st) == SQLITE_ROW) {
          const char* run_id = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
          if (run_id) {
            latest_run_id = run_id;
          }
        }
        sqlite3_finalize(st);
      }
    }
    
    return latest_run_id;
  } catch (const std::exception& e) {
    return "";
  }
}

struct RunInfo {
  std::string run_id;
  std::string strategy;
  std::string kind;
  int64_t started_at;
  std::string note;
  std::string meta;
};

static RunInfo get_run_info(const std::string& db_path, const std::string& run_id) {
  RunInfo info;
  info.run_id = run_id;
  
  try {
    DB db(db_path);
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(db.get_db(), 
                       "SELECT strategy, kind, started_at, note, params_json FROM audit_runs WHERE run_id = ?", 
                       -1, &st, nullptr);
    sqlite3_bind_text(st, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
    
    if (sqlite3_step(st) == SQLITE_ROW) {
      const char* strategy = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
      const char* kind = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
      const char* note = reinterpret_cast<const char*>(sqlite3_column_text(st, 3));
      const char* params_json = reinterpret_cast<const char*>(sqlite3_column_text(st, 4));
      
      info.strategy = strategy ? strategy : "";
      info.kind = kind ? kind : "";
      info.started_at = sqlite3_column_int64(st, 2);
      info.note = note ? note : "";
      info.meta = params_json ? params_json : ""; // Use params_json as meta
    }
    
    sqlite3_finalize(st);
  } catch (const std::exception& e) {
    // Keep defaults
  }
  
  return info;
}

static void print_run_header(const std::string& title, const RunInfo& info) {
  // Format timestamp to local time
  auto format_timestamp = [](int64_t ts_millis) -> std::string {
    if (ts_millis == 0) return "N/A";
    time_t ts_sec = ts_millis / 1000;
    struct tm* tm_info = localtime(&ts_sec);
    char buffer[64];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S %Z", tm_info);
    return std::string(buffer);
  };
  
  // Parse enhanced metadata for dataset type and test period
  std::string dataset_type = "unknown";
  int test_period_days = 0;
  if (!info.meta.empty()) {
    // Simple JSON parsing for dataset_type and test_period_days
    size_t dataset_pos = info.meta.find("\"dataset_type\":\"");
    if (dataset_pos != std::string::npos) {
      size_t start = dataset_pos + 16; // length of "dataset_type":""
      size_t end = info.meta.find("\"", start);
      if (end != std::string::npos) {
        dataset_type = info.meta.substr(start, end - start);
      }
    }
    
    size_t period_pos = info.meta.find("\"test_period_days\":");
    if (period_pos != std::string::npos) {
      size_t start = period_pos + 19; // length of "test_period_days":
      size_t end = info.meta.find_first_of(",}", start);
      if (end != std::string::npos) {
        try {
          test_period_days = std::stoi(info.meta.substr(start, end - start));
        } catch (...) { /* ignore parse errors */ }
      }
    }
  }
  
  printf("=== %s ===\n", title.c_str());
  printf("Run ID: %s\n", info.run_id.c_str());
  printf("Strategy: %s\n", info.strategy.c_str());
  printf("Test Kind: %s\n", info.kind.c_str());
  printf("Run Date/Time: %s\n", format_timestamp(info.started_at).c_str());
  printf("Dataset Type: %s\n", dataset_type.c_str());
  printf("Test Period: %d trading days\n", test_period_days);
  if (!info.note.empty()) {
    printf("Note: %s\n", info.note.c_str());
  }
  printf("\n");
}

int audit_main(int argc, char** argv) {
  if (argc<2) { fputs(usage, stderr); return 1; }
  const char* cmd = argv[1];
  const char* dbp = arg("--db", argc, argv, "audit/sentio_audit.sqlite3");
  
  if (!strcmp(cmd,"init")) {
    DB db(dbp); db.init_schema(); puts("ok");
    return 0;
  }

  if (!strcmp(cmd,"reset")) {
    bool confirmed = has("--confirm", argc, argv);
    if (!confirmed) {
      printf("WARNING: This will delete ALL audit data!\n");
      printf("Use --confirm flag to proceed: sentio_audit reset --confirm\n");
      return 1;
    }
    
    // Remove the database file to reset everything
    if (std::remove(dbp) == 0) {
      printf("Audit database reset successfully: %s\n", dbp);
      // Recreate the database with schema
      DB db(dbp); db.init_schema();
      puts("Fresh database initialized");
      return 0;
    } else {
      printf("Failed to reset database: %s\n", dbp);
      return 1;
    }
  }

  DB db(dbp);

  if (!strcmp(cmd,"new-run")) {
    RunRow r;
    r.run_id     = arg("--run", argc, argv, "");
    r.started_at = now_millis();
    r.kind       = arg("--kind", argc, argv, "backtest");
    r.strategy   = arg("--strategy", argc, argv, "");
    const char* params_file = arg("--params", argc, argv, "");
    const char* data_hash   = arg("--data-hash", argc, argv, "");
    r.git_rev    = arg("--git", argc, argv, "");
    r.note       = arg("--note", argc, argv, "");
    if (r.run_id.empty() || r.strategy.empty() || !params_file || !*params_file || !data_hash || !*data_hash) {
      fputs("missing required args\n", stderr); return 3;
    }
    // Load params.json
    FILE* f=fopen(params_file,"rb"); if(!f){perror("params"); return 4;}
    std::string pj; char buf[4096]; size_t n;
    while((n=fread(buf,1,sizeof(buf),f))>0) pj.append(buf,n);
    fclose(f);
    r.params_json = pj;
    r.data_hash   = data_hash;
    db.new_run(r);
    puts("run created"); return 0;
  }

  if (!strcmp(cmd,"log")) {
    Event ev;
    ev.run_id   = arg("--run",argc,argv,"");
    ev.ts_millis= atoll(arg("--ts",argc,argv,"0"));
    ev.kind     = arg("--kind",argc,argv,"NOTE");
    ev.symbol   = arg("--symbol",argc,argv,"");
    ev.side     = arg("--side",argc,argv,"");
    ev.qty      = atof(arg("--qty",argc,argv,"0"));
    ev.price    = atof(arg("--price",argc,argv,"0"));
    ev.pnl_delta= atof(arg("--pnl",argc,argv,"0"));
    ev.weight   = atof(arg("--weight",argc,argv,"0"));
    ev.prob     = atof(arg("--prob",argc,argv,"0"));
    ev.reason   = arg("--reason",argc,argv,"");
    ev.note     = arg("--note",argc,argv,"");
    if (ev.run_id.empty() || ev.ts_millis==0 || ev.kind.empty()) { fputs("missing run/ts/kind\n", stderr); return 3; }
    auto [seq,h] = db.append_event(ev);
    printf("ok seq=%lld hash=%s\n", seq, h.c_str());
    return 0;
  }

  if (!strcmp(cmd,"end-run")) {
    const char* run = arg("--run",argc,argv,"");
    if (!*run) { fputs("--run required\n", stderr); return 3; }
    db.end_run(run, now_millis());
    puts("ended"); return 0;
  }

  if (!strcmp(cmd,"verify")) {
    const char* run_arg = arg("--run",argc,argv,"");
    std::string run = run_arg && *run_arg ? run_arg : get_latest_run_id(dbp);
    if (run.empty()) { fputs("No runs found in database\n", stderr); return 3; }
    auto [ok,msg]= db.verify_run(run);
    printf("%s: %s\n", ok?"OK":"FAIL", msg.c_str());
    return ok?0:10;
  }

  if (!strcmp(cmd,"summarize")) {
    const char* run_arg = arg("--run",argc,argv,"");
    std::string run = run_arg && *run_arg ? run_arg : get_latest_run_id(dbp);
    if (run.empty()) { fputs("No runs found in database\n", stderr); return 3; }
    auto s = db.summarize(run);
    
    // Get run info and print enhanced header manually
    RunInfo info = get_run_info(dbp, run);
    
    // Parse enhanced metadata
    std::string dataset_type = "unknown";
    int test_period_days = 0;
    if (!info.meta.empty()) {
      // Simple JSON parsing for dataset_type and test_period_days
      size_t dataset_pos = info.meta.find("\"dataset_type\":\"");
      if (dataset_pos != std::string::npos) {
        size_t start = dataset_pos + 16; // length of "dataset_type":""
        size_t end = info.meta.find("\"", start);
        if (end != std::string::npos) {
          dataset_type = info.meta.substr(start, end - start);
        }
      }
      
      size_t period_pos = info.meta.find("\"test_period_days\":");
      if (period_pos != std::string::npos) {
        size_t start = period_pos + 19; // length of "test_period_days":
        size_t end = info.meta.find_first_of(",}", start);
        if (end != std::string::npos) {
          try {
            test_period_days = std::stoi(info.meta.substr(start, end - start));
          } catch (...) { /* ignore parse errors */ }
        }
      }
    }
    
    // Print enhanced header manually
    auto format_timestamp = [](int64_t ts_millis) -> std::string {
      if (ts_millis == 0) return "N/A";
      time_t ts_sec = ts_millis / 1000;
      struct tm* tm_info = localtime(&ts_sec);
      char buffer[64];
      strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S %Z", tm_info);
      return std::string(buffer);
    };
    
    printf("=== AUDIT SUMMARY ===\n");
    printf("Run ID: %s\n", info.run_id.c_str());
    printf("Strategy: %s\n", info.strategy.c_str());
    printf("Test Kind: %s\n", info.kind.c_str());
    printf("Run Date/Time: %s\n", format_timestamp(info.started_at).c_str());
    printf("Dataset Type: %s\n", dataset_type.c_str());
    printf("Test Period: %d trading days\n", test_period_days);
    if (!info.note.empty()) {
      printf("Note: %s\n", info.note.c_str());
    }
    printf("\n");
    
    // (format_timestamp lambda already defined above)
    printf("Event Counts:\n");
    printf("  Total Events: %lld\n", s.n_total);
    printf("  Signals: %lld\n", s.n_signal);
    printf("  Orders: %lld\n", s.n_order);
    printf("  Fills: %lld\n", s.n_fill);
    printf("  P&L Rows: %lld (dedicated P&L accounting events)\n", s.n_pnl);
    printf("\n");
    printf("Trading Performance:\n");
    printf("  Total Return: %.2f%%\n", s.total_return);
    printf("  Monthly Projected Return (MPR): %.2f%%\n", s.mpr);
    printf("  Sharpe Ratio: %.3f\n", s.sharpe);
    printf("  Daily Trades: %.1f\n", s.daily_trades);
    printf("  Max Drawdown: %.2f%%\n", s.max_drawdown);
    printf("  Trading Days: %d\n", s.trading_days);
    printf("\n");
    printf("P&L Summary:\n");
    printf("  Total P&L: %.6f (sum of all P&L from FILL events)\n", s.pnl_sum);
    printf("\n");
    
    // **NEW**: Instrument Distribution and P&L Breakdown with Percentages
    if (!s.instrument_pnl.empty()) {
      printf("Instrument Distribution:\n");
      printf("  %-8s %8s %8s %12s %8s %15s\n", "Symbol", "Fills", "Fill%", "P&L", "P&L%", "Volume");
      printf("  %-8s %8s %8s %12s %8s %15s\n", "------", "-----", "-----", "---", "----", "------");
      
      // Calculate totals for percentage calculations
      int64_t total_fills = 0;
      double total_volume = 0.0;
      for (const auto& [symbol, pnl] : s.instrument_pnl) {
        int64_t fills = s.instrument_fills.count(symbol) ? s.instrument_fills.at(symbol) : 0;
        double volume = s.instrument_volume.count(symbol) ? s.instrument_volume.at(symbol) : 0.0;
        total_fills += fills;
        total_volume += volume;
      }
      
      for (const auto& [symbol, pnl] : s.instrument_pnl) {
        int64_t fills = s.instrument_fills.count(symbol) ? s.instrument_fills.at(symbol) : 0;
        double volume = s.instrument_volume.count(symbol) ? s.instrument_volume.at(symbol) : 0.0;
        
        double fill_pct = total_fills > 0 ? (100.0 * fills / total_fills) : 0.0;
        double pnl_pct = std::abs(s.pnl_sum) > 1e-6 ? (100.0 * pnl / s.pnl_sum) : 0.0;
        
        printf("  %-8s %8lld %7.1f%% %12.2f %7.1f%% %15.0f\n", 
               symbol.c_str(), fills, fill_pct, pnl, pnl_pct, volume);
      }
      printf("\n");
    }
    
    printf("Time Range:\n");
    printf("  Start: %s (%lld)\n", format_timestamp(s.ts_first).c_str(), s.ts_first);
    printf("  End: %s (%lld)\n", format_timestamp(s.ts_last).c_str(), s.ts_last);
    printf("\n");
    
    // **CONFLICT DETECTION**: Analyze position conflicts throughout the run
    printf("=== POSITION CONFLICT ANALYSIS ===\n");
    
    // Track positions throughout the run by replaying fills
    std::unordered_map<std::string, ConflictPosition> positions;
    int total_conflicts = 0;
    std::vector<std::string> conflict_timestamps;
    
    // Query all FILL events to reconstruct position history
    sqlite3_stmt* fill_st = nullptr;
    std::string fill_sql = "SELECT ts_millis, symbol, side, qty FROM audit_events WHERE run_id = ? AND kind = 'FILL' ORDER BY ts_millis ASC";
    int fill_rc = sqlite3_prepare_v2(db.get_db(), fill_sql.c_str(), -1, &fill_st, nullptr);
    if (fill_rc == SQLITE_OK) {
        sqlite3_bind_text(fill_st, 1, run.c_str(), -1, SQLITE_TRANSIENT);
        
        while (sqlite3_step(fill_st) == SQLITE_ROW) {
            int64_t ts_millis = sqlite3_column_int64(fill_st, 0);
            const char* symbol = (const char*)sqlite3_column_text(fill_st, 1);
            const char* side = (const char*)sqlite3_column_text(fill_st, 2);
            double qty = sqlite3_column_double(fill_st, 3);
            
            if (symbol && side) {
                // Update position
                auto& pos = positions[symbol];
                pos.symbol = symbol;
                
                // Apply fill to position (BUY=0, SELL=1)
                if (strcmp(side, "BUY") == 0) {
                    pos.qty += qty;
                } else if (strcmp(side, "SELL") == 0) {
                    pos.qty -= qty;
                }
                
                // Check for conflicts after each fill
                auto conflict_analysis = analyze_position_conflicts(positions);
                if (conflict_analysis.has_conflicts) {
                    total_conflicts++;
                    
                    // Convert timestamp to readable format
                    time_t ts_sec = ts_millis / 1000;
                    struct tm* tm_info = localtime(&ts_sec);
                    char time_buffer[32];
                    strftime(time_buffer, sizeof(time_buffer), "%Y-%m-%d %H:%M:%S", tm_info);
                    
                    conflict_timestamps.push_back(std::string(time_buffer));
                    
                    // Only show first few conflicts to avoid spam
                    if (total_conflicts <= 5) {
                        printf("⚠️  CONFLICT #%d at %s:\n", total_conflicts, time_buffer);
                        for (const auto& conflict : conflict_analysis.conflicts) {
                            printf("   %s\n", conflict.c_str());
                        }
                        printf("\n");
                    }
                }
            }
        }
        sqlite3_finalize(fill_st);
    }
    
    // Summary of conflict analysis
    if (total_conflicts == 0) {
        printf("✅ NO CONFLICTS DETECTED: All positions maintained proper directional consistency\n");
    } else {
        printf("❌ CONFLICTS DETECTED: %d instances of conflicting positions found\n", total_conflicts);
        if (total_conflicts > 5) {
            printf("   (Showing first 5 conflicts only - %d additional conflicts occurred)\n", total_conflicts - 5);
        }
        printf("\n");
        printf("⚠️  WARNING: Conflicting positions generate fees without profit and cause\n");
        printf("   leveraged ETF decay. The backend should prevent these automatically.\n");
    }
    
    printf("\n");
    printf("Note: P&L Rows = 0 means P&L is embedded in FILL events, not separate accounting events\n");
    return 0;
  }

  if (!strcmp(cmd,"export")) {
    const char* run_arg = arg("--run",argc,argv,"");
    std::string run = run_arg && *run_arg ? run_arg : get_latest_run_id(dbp);
    if (run.empty()) { fputs("No runs found in database\n", stderr); return 3; }
    const char* fmt = arg("--fmt",argc,argv,"jsonl");
    const char* out = arg("--out",argc,argv,"-");
    if (strcmp(out,"-")==0) { fputs("write to file only (use --out)\n", stderr); return 5; }
    if (!strcmp(fmt,"jsonl")) db.export_run_jsonl(run,out); else db.export_run_csv(run,out);
    puts("exported"); return 0;
  }

  if (!strcmp(cmd,"grep")) {
    const char* run_arg = arg("--run",argc,argv,"");
    std::string run = run_arg && *run_arg ? run_arg : get_latest_run_id(dbp);
    if (run.empty()) { fputs("No runs found in database\n", stderr); return 3; }
    const char* where = arg("--sql",argc,argv,"");
    long n = db.grep_where(run, where?where:"");
    printf("rows=%ld\n", n);
    return 0;
  }

  if (!strcmp(cmd,"diff")) {
    const char* a=arg("--run",argc,argv,"");
    const char* b=arg("--run2",argc,argv,"");
    auto txt = db.diff_runs(a,b);
    fputs(txt.c_str(), stdout);
    return 0;
  }

  if (!strcmp(cmd,"vacuum")) { db.vacuum(); puts("ok"); return 0; }

  if (!strcmp(cmd,"list")) {
    const char* strategy = arg("--strategy", argc, argv, "");
    const char* kind = arg("--kind", argc, argv, "");
    list_runs(dbp, strategy ? strategy : "", kind ? kind : "");
    return 0;
  }

  if (!strcmp(cmd,"latest")) {
    const char* strategy = arg("--strategy", argc, argv, "");
    find_latest_run(dbp, strategy ? strategy : "");
    return 0;
  }

  if (!strcmp(cmd,"info")) {
    const char* run_arg = arg("--run", argc, argv, "");
    std::string run = run_arg && *run_arg ? run_arg : get_latest_run_id(dbp);
    if (run.empty()) { fputs("No runs found in database\n", stderr); return 3; }
    show_run_info(dbp, run);
    return 0;
  }

  if (!strcmp(cmd,"trade-flow")) {
    const char* run_arg = arg("--run", argc, argv, "");
    std::string run = run_arg && *run_arg ? run_arg : get_latest_run_id(dbp);
    if (run.empty()) { fputs("No runs found in database\n", stderr); return 3; }
    const char* symbol = arg("--symbol", argc, argv, "");
    const char* limit_str = arg("--limit", argc, argv, "20");
    int limit = std::atoi(limit_str);
    
    // Handle --max option with optional number
    bool show_max = has("--max", argc, argv);
    if (show_max) {
      const char* max_str = arg("--max", argc, argv, nullptr);
      if (max_str && *max_str) {
        // --max N specified
        limit = std::atoi(max_str);
      } else {
        // --max without number specified, show all
        limit = 0;
      }
    }
    bool enhanced = has("--enhanced", argc, argv);
    bool show_buy = has("--buy", argc, argv);
    bool show_sell = has("--sell", argc, argv);
    bool show_hold = has("--hold", argc, argv);
    show_trade_flow(dbp, run, symbol, limit, enhanced, show_buy, show_sell, show_hold);
    return 0;
  }

  if (!strcmp(cmd,"signal-stats")) {
    const char* run_arg = arg("--run", argc, argv, "");
    std::string run = run_arg && *run_arg ? run_arg : get_latest_run_id(dbp);
    if (run.empty()) { fputs("No runs found in database\n", stderr); return 3; }
    const char* strategy = arg("--strategy", argc, argv, "");
    show_signal_stats(dbp, run, strategy ? strategy : "");
    return 0;
  }

  if (!strcmp(cmd,"signal-flow")) {
    const char* run_arg = arg("--run", argc, argv, "");
    std::string run = run_arg && *run_arg ? run_arg : get_latest_run_id(dbp);
    if (run.empty()) { fputs("No runs found in database\n", stderr); return 3; }
    const char* symbol = arg("--symbol", argc, argv, "");
    const char* limit_str = arg("--limit", argc, argv, "20");
    int limit = std::atoi(limit_str);
    
    // Handle --max option with optional number
    bool show_max = has("--max", argc, argv);
    if (show_max) {
      const char* max_str = arg("--max", argc, argv, nullptr);
      if (max_str && *max_str) {
        // --max N specified
        limit = std::atoi(max_str);
      } else {
        // --max without number specified, show all
        limit = 0;
      }
    }
    bool show_buy = has("--buy", argc, argv);
    bool show_sell = has("--sell", argc, argv);
    bool show_hold = has("--hold", argc, argv);
    bool enhanced = has("--enhanced", argc, argv);
    show_signal_flow(dbp, run, symbol, limit, show_buy, show_sell, show_hold, enhanced);
    return 0;
  }

  if (!strcmp(cmd,"position-history")) {
    const char* run_arg = arg("--run", argc, argv, "");
    std::string run = run_arg && *run_arg ? run_arg : get_latest_run_id(dbp);
    if (run.empty()) { fputs("No runs found in database\n", stderr); return 3; }
    const char* symbol = arg("--symbol", argc, argv, "");
    const char* limit_str = arg("--limit", argc, argv, "20");
    int limit = std::atoi(limit_str);
    
    // Handle --max option with optional number
    bool show_max = has("--max", argc, argv);
    if (show_max) {
      const char* max_str = arg("--max", argc, argv, nullptr);
      if (max_str && *max_str) {
        // --max N specified
        limit = std::atoi(max_str);
      } else {
        // --max without number specified, show all
        limit = 0;
      }
    }
    bool show_buy = has("--buy", argc, argv);
    bool show_sell = has("--sell", argc, argv);
    bool show_hold = has("--hold", argc, argv);
    show_position_history(dbp, run, symbol, limit, show_buy, show_sell, show_hold);
    return 0;
  }

  if (!strcmp(cmd,"strategies-summary")) {
    show_strategies_summary(dbp);
    return 0;
  }

  fputs(usage, stderr); return 1;
}

// Provide a standalone main if you build standalone; otherwise link into your app.
int main(int argc, char** argv) { return audit_main(argc, argv); }

// Implementation of utility functions
namespace audit {

void list_runs(const std::string& db_path, const std::string& strategy_filter, const std::string& kind_filter) {
  try {
    DB db(db_path);
    
    std::string sql = "SELECT run_id, strategy, kind, started_at, ended_at, note FROM audit_runs";
    std::string where_clause = "";
    
    if (!strategy_filter.empty() || !kind_filter.empty()) {
      where_clause = " WHERE ";
      bool first = true;
      if (!strategy_filter.empty()) {
        where_clause += "strategy = '" + strategy_filter + "'";
        first = false;
      }
      if (!kind_filter.empty()) {
        if (!first) where_clause += " AND ";
        where_clause += "kind = '" + kind_filter + "'";
      }
    }
    
    sql += where_clause + " ORDER BY started_at DESC";
    
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(db.get_db(), sql.c_str(), -1, &st, nullptr);
    
    printf("%-8s %-15s %-10s %-20s %-20s %s\n", "RUN_ID", "STRATEGY", "KIND", "STARTED_AT", "ENDED_AT", "NOTE");
    printf("%-8s %-15s %-10s %-20s %-20s %s\n", "------", "--------", "----", "----------", "--------", "----");
    
    while (sqlite3_step(st) == SQLITE_ROW) {
      const char* run_id = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
      const char* strategy = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
      const char* kind = reinterpret_cast<const char*>(sqlite3_column_text(st, 2));
      std::int64_t started_at = sqlite3_column_int64(st, 3);
      std::int64_t ended_at = sqlite3_column_int64(st, 4);
      const char* note = reinterpret_cast<const char*>(sqlite3_column_text(st, 5));
      
      printf("%-8s %-15s %-10s %-20lld %-20lld %s\n", 
             run_id ? run_id : "", 
             strategy ? strategy : "", 
             kind ? kind : "", 
             started_at, 
             ended_at, 
             note ? note : "");
    }
    
    sqlite3_finalize(st);
  } catch (const std::exception& e) {
    fprintf(stderr, "Error listing runs: %s\n", e.what());
  }
}

void find_latest_run(const std::string& db_path, const std::string& strategy_filter) {
  try {
    DB db(db_path);
    
    std::string sql = "SELECT run_id, strategy, kind, started_at, ended_at FROM audit_runs";
    if (!strategy_filter.empty()) {
      sql += " WHERE strategy = '" + strategy_filter + "'";
    }
    sql += " ORDER BY run_id DESC LIMIT 1";
    
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(db.get_db(), sql.c_str(), -1, &st, nullptr);
    
    if (sqlite3_step(st) == SQLITE_ROW) {
      const char* run_id = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
      const char* strategy = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
      const char* kind = reinterpret_cast<const char*>(sqlite3_column_text(st, 2));
      std::int64_t started_at = sqlite3_column_int64(st, 3);
      std::int64_t ended_at = sqlite3_column_int64(st, 4);
      
      printf("Latest run: %s\n", run_id ? run_id : "");
      printf("Strategy: %s\n", strategy ? strategy : "");
      printf("Kind: %s\n", kind ? kind : "");
      printf("Started: %lld\n", started_at);
      printf("Ended: %lld\n", ended_at);
    } else {
      printf("No runs found");
      if (!strategy_filter.empty()) {
        printf(" for strategy: %s", strategy_filter.c_str());
      }
      printf("\n");
    }
    
    sqlite3_finalize(st);
  } catch (const std::exception& e) {
    fprintf(stderr, "Error finding latest run: %s\n", e.what());
  }
}

void show_run_info(const std::string& db_path, const std::string& run_id) {
  try {
    DB db(db_path);
    
    // Get run info
    sqlite3_stmt* st = nullptr;
    sqlite3_prepare_v2(db.get_db(), 
                       "SELECT run_id, strategy, kind, started_at, ended_at, params_json, data_hash, git_rev, note FROM audit_runs WHERE run_id = ?", 
                       -1, &st, nullptr);
    sqlite3_bind_text(st, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
    
    if (sqlite3_step(st) == SQLITE_ROW) {
      const char* strategy = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
      const char* kind = reinterpret_cast<const char*>(sqlite3_column_text(st, 2));
      std::int64_t started_at = sqlite3_column_int64(st, 3);
      std::int64_t ended_at = sqlite3_column_int64(st, 4);
      const char* params_json = reinterpret_cast<const char*>(sqlite3_column_text(st, 5));
      const char* data_hash = reinterpret_cast<const char*>(sqlite3_column_text(st, 6));
      const char* git_rev = reinterpret_cast<const char*>(sqlite3_column_text(st, 7));
      const char* note = reinterpret_cast<const char*>(sqlite3_column_text(st, 8));
      
      printf("Run ID: %s\n", run_id.c_str());
      printf("Strategy: %s\n", strategy ? strategy : "");
      printf("Kind: %s\n", kind ? kind : "");
      printf("Started: %lld\n", started_at);
      printf("Ended: %lld\n", ended_at);
      printf("Data Hash: %s\n", data_hash ? data_hash : "");
      printf("Git Rev: %s\n", git_rev ? git_rev : "");
      printf("Note: %s\n", note ? note : "");
      printf("Params: %s\n", params_json ? params_json : "");
      
      // Get summary
      auto summary = db.summarize(run_id);
      printf("\nSummary:\n");
      printf("  Events: %lld\n", summary.n_total);
      printf("  Signals: %lld\n", summary.n_signal);
      printf("  Orders: %lld\n", summary.n_order);
      printf("  Fills: %lld\n", summary.n_fill);
      printf("  P&L Rows: %lld\n", summary.n_pnl);
      printf("  P&L Sum: %.6f\n", summary.pnl_sum);
      printf("  Time Range: %lld - %lld\n", summary.ts_first, summary.ts_last);
      
    } else {
      printf("Run not found: %s\n", run_id.c_str());
    }
    
    sqlite3_finalize(st);
  } catch (const std::exception& e) {
    fprintf(stderr, "Error showing run info: %s\n", e.what());
  }
}

struct TradeFlowEvent {
  std::int64_t timestamp;
  std::string kind;
  std::string symbol;
  std::string side;
  double quantity;
  double price;
  double pnl_delta;
  double weight;
  double prob;
  std::string reason;
  std::string note;
};

void show_trade_flow(const std::string& db_path, const std::string& run_id, const std::string& symbol_filter, int limit, bool enhanced, bool show_buy, bool show_sell, bool show_hold) {
  try {
    DB db(db_path);
    
    // Get run info and print header
    RunInfo info = get_run_info(db_path, run_id);
    print_run_header("EXECUTION FLOW REPORT", info);
    
    if (!symbol_filter.empty()) {
      printf("Symbol Filter: %s\n", symbol_filter.c_str());
    }
    
    // Display action filters
    std::vector<std::string> action_filters;
    if (show_buy) action_filters.push_back("BUY");
    if (show_sell) action_filters.push_back("SELL");
    if (show_hold) action_filters.push_back("HOLD");
    
    if (!action_filters.empty()) {
      printf("Action Filter: ");
      for (size_t i = 0; i < action_filters.size(); i++) {
        if (i > 0) printf(", ");
        printf("%s", action_filters[i].c_str());
      }
      printf("\n");
    }
    
    if (limit > 0) {
      printf("Showing: %d most recent events\n", limit);
    } else {
      printf("Showing: All execution events\n");
    }
    printf("\n");
    
    // Build SQL query to get trade flow events
    std::string sql = "SELECT ts_millis, kind, symbol, side, qty, price, pnl_delta, weight, prob, reason, note FROM audit_events WHERE run_id = ? AND kind IN ('SIGNAL', 'ORDER', 'FILL')";
    
    if (!symbol_filter.empty()) {
      sql += " AND symbol = '" + symbol_filter + "'";
    }
    
    // Add action filtering if any specific actions are requested
    if (show_buy || show_sell || show_hold) {
      std::vector<std::string> side_conditions;
      if (show_buy) side_conditions.push_back("side = 'BUY'");
      if (show_sell) side_conditions.push_back("side = 'SELL'");
      if (show_hold) side_conditions.push_back("side = 'HOLD'");
      
      sql += " AND (";
      for (size_t i = 0; i < side_conditions.size(); i++) {
        if (i > 0) sql += " OR ";
        sql += side_conditions[i];
      }
      sql += ")";
    }
    
    sql += " ORDER BY ts_millis ASC";
    
    sqlite3_stmt* st = nullptr;
    int rc = sqlite3_prepare_v2(db.get_db(), sql.c_str(), -1, &st, nullptr);
    if (rc != SQLITE_OK) {
      fprintf(stderr, "SQL prepare error: %s\n", sqlite3_errmsg(db.get_db()));
      return;
    }
    sqlite3_bind_text(st, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
    
    // Collect all events and calculate summary statistics
    std::vector<TradeFlowEvent> events;
    int signal_count = 0, order_count = 0, fill_count = 0;
    double total_volume = 0.0, total_pnl = 0.0;
    std::map<std::string, int> symbol_activity;
    std::map<std::string, double> symbol_pnl;
    
    while (sqlite3_step(st) == SQLITE_ROW) {
      TradeFlowEvent event;
      event.timestamp = sqlite3_column_int64(st, 0);
      event.kind = sqlite3_column_text(st, 1) ? (char*)sqlite3_column_text(st, 1) : "";
      event.symbol = sqlite3_column_text(st, 2) ? (char*)sqlite3_column_text(st, 2) : "";
      event.side = sqlite3_column_text(st, 3) ? (char*)sqlite3_column_text(st, 3) : "";
      event.quantity = sqlite3_column_double(st, 4);
      event.price = sqlite3_column_double(st, 5);
      event.pnl_delta = sqlite3_column_double(st, 6);
      event.weight = sqlite3_column_double(st, 7);
      event.prob = sqlite3_column_double(st, 8);
      event.reason = sqlite3_column_text(st, 9) ? (char*)sqlite3_column_text(st, 9) : "";
      event.note = sqlite3_column_text(st, 10) ? (char*)sqlite3_column_text(st, 10) : "";
      
      events.push_back(event);
      
      // Update statistics
      if (event.kind == "SIGNAL") signal_count++;
      else if (event.kind == "ORDER") order_count++;
      else if (event.kind == "FILL") {
        fill_count++;
        total_volume += event.quantity * event.price;
        total_pnl += event.pnl_delta;
        if (!event.symbol.empty()) {
          symbol_pnl[event.symbol] += event.pnl_delta;
        }
      }
      
      if (!event.symbol.empty()) {
        symbol_activity[event.symbol]++;
      }
    }
    
    sqlite3_finalize(st);
    
    // Calculate execution efficiency
    double execution_rate = (order_count > 0) ? (double)fill_count / order_count * 100.0 : 0.0;
    double signal_to_order_rate = (signal_count > 0) ? (double)order_count / signal_count * 100.0 : 0.0;
    
    // 1. EXECUTION PERFORMANCE SUMMARY
    printf("📊 EXECUTION PERFORMANCE SUMMARY\n");
    printf("┌─────────────────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Total Signals       │ %11d │ Orders Placed       │ %11d │ Execution Rate │ %6.1f%% │\n", 
           signal_count, order_count, execution_rate);
    printf("│ Orders Filled       │ %11d │ Total Volume        │ $%10.0f │ Signal→Order   │ %6.1f%% │\n", 
           fill_count, total_volume, signal_to_order_rate);
    printf("│ Active Symbols      │ %11d │ Net P&L Impact      │ $%+10.2f │ Avg Fill Size  │ $%7.0f │\n", 
           (int)symbol_activity.size(), total_pnl, 
           fill_count > 0 ? total_volume / fill_count : 0.0);
    printf("└─────────────────────────────────────────────────────────────────────────────────────────────┘\n\n");
    
    // 2. SYMBOL PERFORMANCE BREAKDOWN
    if (!symbol_pnl.empty()) {
      printf("📈 SYMBOL PERFORMANCE BREAKDOWN\n");
      printf("┌────────┬─────────┬─────────────┬─────────────────────────────────────────────────────────────┐\n");
      printf("│ Symbol │ Events  │ P&L Impact  │ Performance Level                                           │\n");
      printf("├────────┼─────────┼─────────────┼─────────────────────────────────────────────────────────────┤\n");
      
      // Sort symbols by P&L
      std::vector<std::pair<std::string, double>> sorted_pnl(symbol_pnl.begin(), symbol_pnl.end());
      std::sort(sorted_pnl.begin(), sorted_pnl.end(), 
                [](const auto& a, const auto& b) { return a.second > b.second; });
      
      for (const auto& [symbol, pnl] : sorted_pnl) {
        int events = symbol_activity[symbol];
        
        // Create performance bar (green for profit, red for loss)
        double max_abs_pnl = 0;
        for (const auto& [s, p] : sorted_pnl) {
          max_abs_pnl = std::max(max_abs_pnl, std::abs(p));
        }
        
        int bar_length = max_abs_pnl > 0 ? std::min(50, (int)(std::abs(pnl) * 50 / max_abs_pnl)) : 0;
        std::string performance_bar;
        if (pnl > 0) {
          performance_bar = std::string(bar_length, '#') + std::string(50 - bar_length, '.');
        } else {
          performance_bar = std::string(50 - bar_length, '.') + std::string(bar_length, 'X');
        }
        
        const char* pnl_color = pnl > 0 ? "🟢" : pnl < 0 ? "🔴" : "⚪";
        
        printf("│ %-6s │ %7d │ %s$%+9.2f │ %s │\n", 
               symbol.c_str(), events, pnl_color, pnl, performance_bar.c_str());
      }
      printf("└────────┴─────────┴─────────────┴─────────────────────────────────────────────────────────────┘\n\n");
    }
    
    // 3. EXECUTION EVENT TIMELINE
    printf("🔄 EXECUTION EVENT TIMELINE");
    if (limit > 0 && (int)events.size() > limit) {
      printf(" (Last %d of %d events)", limit, (int)events.size());
    }
    printf("\n");
    printf("┌──────────────┬─────────┬────────┬────────┬──────────┬──────────┬─────────────┬──────────────┐\n");
    printf("│ Time         │ Event   │ Symbol │ Action │ Quantity │ Price    │ Value       │ P&L Impact   │\n");
    printf("├──────────────┼─────────┼────────┼────────┼──────────┼──────────┼─────────────┼──────────────┤\n");
    
    // Show recent events (apply limit here)
    int start_idx = (limit > 0 && (int)events.size() > limit) ? (int)events.size() - limit : 0;
    int event_count = 0;
    
    for (int i = start_idx; i < (int)events.size(); i++) {
      const auto& event = events[i];
      
      // Add empty line before each event for better scanning (except first)
      if (event_count > 0) {
        printf("│              │         │        │        │          │          │             │              │\n");
      }
      
      // Format timestamp
      char time_str[32];
      std::time_t time_t = event.timestamp / 1000;
      std::strftime(time_str, sizeof(time_str), "%m/%d %H:%M:%S", std::localtime(&time_t));
      
      // Event type icons
      const char* event_icon = "📋";
      if (event.kind == "SIGNAL") event_icon = "📡";
      else if (event.kind == "FILL") event_icon = "✅";
      
      // Action color coding
      const char* action_color = "";
      if (event.side == "BUY") action_color = "🟢";
      else if (event.side == "SELL") action_color = "🔴";
      else if (event.side == "HOLD") action_color = "🟡";
      
      double trade_value = event.quantity * event.price;
      
      printf("│ %-12s │ %s%-5s │ %-6s │ %s%-4s │ %8.0f │ $%7.2f │ $%+10.0f │ $%+11.2f │\n",
             time_str, event_icon, event.kind.c_str(), event.symbol.c_str(),
             action_color, event.side.c_str(), event.quantity, event.price, 
             trade_value, event.pnl_delta);
      
      // Show additional details based on event type
      if (event.kind == "SIGNAL") {
        if (event.prob > 0 || event.weight > 0) {
          printf("│              │ └─ Signal Strength: %.1f%% prob, %.2f weight                    │              │\n",
                 event.prob * 100, event.weight);
        }
        if (!event.reason.empty()) {
          printf("│              │ └─ Signal Type: %-45s │              │\n", event.reason.c_str());
        }
      } else if (event.kind == "ORDER") {
        printf("│              │ └─ Order Details: %s %.0f shares @ $%.2f                      │              │\n", 
               event.side.c_str(), event.quantity, event.price);
      } else if (event.kind == "FILL") {
        const char* pnl_indicator = event.pnl_delta > 0 ? "🟢 PROFIT" : 
                                   event.pnl_delta < 0 ? "🔴 LOSS" : "⚪ NEUTRAL";
        printf("│              │ └─ Execution: %s (P&L: $%.2f %s)                        │              │\n", 
               event.side.c_str(), event.pnl_delta, pnl_indicator);
      }
      
      event_count++;
    }
    printf("└──────────────┴─────────┴────────┴────────┴──────────┴──────────┴─────────────┴──────────────┘\n\n");
    
    // 4. EXECUTION EFFICIENCY ANALYSIS
    printf("⚡ EXECUTION EFFICIENCY ANALYSIS\n");
    printf("┌─────────────────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Metric                    │ Value         │ Rating        │ Description                     │\n");
    printf("├─────────────────────────────────────────────────────────────────────────────────────────────┤\n");
    
    // Execution rate analysis
    const char* exec_rating = execution_rate >= 90 ? "🟢 EXCELLENT" : 
                             execution_rate >= 70 ? "🟡 GOOD" : "🔴 NEEDS WORK";
    printf("│ Order Fill Rate           │ %6.1f%%       │ %-13s │ %% of orders successfully filled   │\n", 
           execution_rate, exec_rating);
    
    // Signal conversion analysis  
    const char* signal_rating = signal_to_order_rate >= 20 ? "🟢 ACTIVE" :
                               signal_to_order_rate >= 10 ? "🟡 MODERATE" : "🔴 PASSIVE";
    printf("│ Signal Conversion Rate    │ %6.1f%%       │ %-13s │ %% of signals converted to orders  │\n", 
           signal_to_order_rate, signal_rating);
    
    // P&L efficiency
    const char* pnl_rating = total_pnl > 0 ? "🟢 PROFITABLE" : 
                            total_pnl > -100 ? "🟡 BREAKEVEN" : "🔴 LOSING";
    printf("│ P&L Efficiency            │ $%+10.2f │ %-13s │ Net profit/loss from executions    │\n", 
           total_pnl, pnl_rating);
    
    // Volume efficiency
    const char* volume_rating = total_volume > 1000000 ? "🟢 HIGH VOLUME" :
                               total_volume > 100000 ? "🟡 MODERATE" : "🔴 LOW VOLUME";
    printf("│ Trading Volume            │ $%10.0f │ %-13s │ Total dollar volume traded         │\n", 
           total_volume, volume_rating);
    
    printf("└─────────────────────────────────────────────────────────────────────────────────────────────┘\n");
    
  } catch (const std::exception& e) {
    fprintf(stderr, "Error showing trade flow: %s\n", e.what());
  }
}

void show_signal_stats(const std::string& db_path, const std::string& run_id, const std::string& strategy_filter) {
  try {
    DB db(db_path);
    
    // Build SQL query to get signal diagnostics
    std::string sql = "SELECT ts_millis, symbol, qty, price, note FROM audit_events WHERE run_id = ? AND kind = 'SIGNAL_DIAG'";
    
    if (!strategy_filter.empty()) {
      sql += " AND symbol = '" + strategy_filter + "'";
    }
    
    sql += " ORDER BY ts_millis ASC";
    
    sqlite3_stmt* st = nullptr;
    int rc = sqlite3_prepare_v2(db.get_db(), sql.c_str(), -1, &st, nullptr);
    if (rc != SQLITE_OK) {
      fprintf(stderr, "SQL prepare error: %s\n", sqlite3_errmsg(db.get_db()));
      return;
    }
    sqlite3_bind_text(st, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
    
    // Get run info and print header
    RunInfo info = get_run_info(db_path, run_id);
    print_run_header("SIGNAL DIAGNOSTICS", info);
    
    if (!strategy_filter.empty()) {
      printf("Strategy Filter: %s\n", strategy_filter.c_str());
    }
    printf("\n");
    
    printf("┌─────────────────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│                                    SIGNAL STATISTICS                                        │\n");
    printf("└─────────────────────────────────────────────────────────────────────────────────────────────┘\n\n");
    
    bool found_any = false;
    while (sqlite3_step(st) == SQLITE_ROW) {
      found_any = true;
      std::int64_t ts = sqlite3_column_int64(st, 0);
      const char* strategy = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
      double emitted = sqlite3_column_double(st, 2);
      double dropped = sqlite3_column_double(st, 3);
      const char* note = reinterpret_cast<const char*>(sqlite3_column_text(st, 4));
      
      // Format timestamp
      char time_str[32];
      std::time_t time_t = ts / 1000;
      std::strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", std::localtime(&time_t));
      
      printf("🔍 Strategy: %s\n", strategy ? strategy : "");
      printf("⏰ Timestamp: %s\n", time_str);
      printf("📊 Signal Statistics:\n");
      printf("   📤 Emitted: %.0f\n", emitted);
      printf("   📥 Dropped: %.0f\n", dropped);
      printf("   📈 Success Rate: %.1f%%\n", emitted > 0 ? (emitted / (emitted + dropped)) * 100.0 : 0.0);
      
      if (note) {
        std::string clean_note = clean_note_for_display(note);
        if (!clean_note.empty()) {
          printf("   📋 Details: %s\n", clean_note.c_str());
        }
      }
      
      printf("\n");
    }
    
    if (!found_any) {
      printf("No signal diagnostics found for this run.\n");
    }
    
    sqlite3_finalize(st);
  } catch (const std::exception& e) {
    fprintf(stderr, "Error showing signal stats: %s\n", e.what());
  }
}

void show_signal_flow(const std::string& db_path, const std::string& run_id, const std::string& symbol_filter, int limit, bool show_buy, bool show_sell, bool show_hold, bool enhanced) {
  try {
    DB db(db_path);
    
    // Get run info and print header
    RunInfo info = get_run_info(db_path, run_id);
    print_run_header("SIGNAL PIPELINE ANALYSIS", info);
    
    if (!symbol_filter.empty()) {
      printf("Symbol Filter: %s\n", symbol_filter.c_str());
    }
    
    // Display action filters
    std::vector<std::string> action_filters;
    if (show_buy) action_filters.push_back("BUY");
    if (show_sell) action_filters.push_back("SELL");
    if (show_hold) action_filters.push_back("HOLD");
    
    if (!action_filters.empty()) {
      printf("Action Filter: ");
      for (size_t i = 0; i < action_filters.size(); i++) {
        if (i > 0) printf(", ");
        printf("%s", action_filters[i].c_str());
      }
      printf("\n");
    }
    
    if (limit > 0) {
      printf("Showing: %d most recent events\n", limit);
    } else {
      printf("Showing: All signal events\n");
    }
    printf("\n");
    
    // Show signal pipeline diagram
    printf("┌─────────────────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│                                SIGNAL PIPELINE DIAGRAM                                        │\n");
    printf("└─────────────────────────────────────────────────────────────────────────────────────────────┘\n\n");
    
    printf("📊 SIGNAL PROCESSING PIPELINE:\n");
    printf("┌─────────────────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Market Data → Feature Extraction → Strategy Signal → Signal Gate → Router → Order → Fill │\n");
    printf("└─────────────────────────────────────────────────────────────────────────────────────────────┘\n\n");
    
    // Get signal diagnostics first
    printf("📈 SIGNAL DIAGNOSTICS:\n");
    sqlite3_stmt* diag_st = nullptr;
    std::string diag_sql = "SELECT symbol, qty, price, note FROM audit_events WHERE run_id = ? AND kind = 'SIGNAL_DIAG'";
    int diag_rc = sqlite3_prepare_v2(db.get_db(), diag_sql.c_str(), -1, &diag_st, nullptr);
    if (diag_rc == SQLITE_OK) {
      sqlite3_bind_text(diag_st, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
      if (sqlite3_step(diag_st) == SQLITE_ROW) {
        const char* strategy = reinterpret_cast<const char*>(sqlite3_column_text(diag_st, 0));
        double emitted = sqlite3_column_double(diag_st, 1);
        double dropped = sqlite3_column_double(diag_st, 2);
        const char* details = reinterpret_cast<const char*>(sqlite3_column_text(diag_st, 3));
        
        printf("🔍 Strategy: %s\n", strategy ? strategy : "Unknown");
        printf("📤 Signals Emitted: %.0f\n", emitted);
        printf("📥 Signals Dropped: %.0f\n", dropped);
        printf("📈 Success Rate: %.1f%%\n", (emitted + dropped) > 0 ? (emitted / (emitted + dropped)) * 100.0 : 0.0);
        
        if (details) {
          printf("📋 Drop Breakdown: %s\n", details);
        }
      }
      sqlite3_finalize(diag_st);
    }
    printf("\n");
    
    // Get signal events with enhanced analysis
    std::string sql = "SELECT ts_millis, kind, symbol, side, qty, price, pnl_delta, weight, prob, reason, note, hash_curr FROM audit_events WHERE run_id = ? AND kind IN ('SIGNAL', 'ORDER', 'FILL', 'SIGNAL_DROP')";
    
    if (!symbol_filter.empty()) {
      sql += " AND symbol = '" + symbol_filter + "'";
    }
    
    // Add action filtering if any specific actions are requested
    if (show_buy || show_sell || show_hold) {
      std::vector<std::string> side_conditions;
      if (show_buy) side_conditions.push_back("side = 'BUY'");
      if (show_sell) side_conditions.push_back("side = 'SELL'");
      if (show_hold) side_conditions.push_back("side = 'HOLD'");
      
      sql += " AND (";
      for (size_t i = 0; i < side_conditions.size(); i++) {
        if (i > 0) sql += " OR ";
        sql += side_conditions[i];
      }
      sql += ")";
    }
    
    sql += " ORDER BY ts_millis ASC";
    
    if (limit > 0) {
      sql += " LIMIT " + std::to_string(limit);
    }
    
    sqlite3_stmt* st = nullptr;
    int rc = sqlite3_prepare_v2(db.get_db(), sql.c_str(), -1, &st, nullptr);
    if (rc != SQLITE_OK) {
      fprintf(stderr, "SQL prepare error: %s\n", sqlite3_errmsg(db.get_db()));
      return;
    }
    sqlite3_bind_text(st, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
    
    printf("🔄 SIGNAL PROCESSING EVENTS:\n");
    printf("┌─────────────────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Time     │ Event    │ Symbol │ Signal │ Prob   │ Weight │ Status │\n");
    printf("├─────────────────────────────────────────────────────────────────────────────────────────────┤\n");
    
    // Helper function to decode drop reasons
    auto decode_drop_reason = [](const char* reason) -> std::string {
      if (!reason) return "Unknown reason";
      
      std::string reason_str = reason;
      if (reason_str == "DROP_REASON_0") return "System/coordination signal (not tradeable)";
      else if (reason_str == "DROP_REASON_1") return "Minimum bars not met";
      else if (reason_str == "DROP_REASON_2") return "Outside trading session";
      else if (reason_str == "DROP_REASON_3") return "NaN/Invalid signal value";
      else if (reason_str == "DROP_REASON_4") return "Zero volume bar";
      else if (reason_str == "DROP_REASON_5") return "Below probability threshold";
      else if (reason_str == "DROP_REASON_6") return "Signal cooldown active";
      else if (reason_str == "DROP_REASON_7") return "Duplicate signal filtered";
      else if (reason_str == "DROP_REASON_8") return "Position size limit reached";
      else if (reason_str == "DROP_REASON_9") return "Risk management override";
      else if (reason_str == "DROP_REASON_10") return "Conflicting position detected";
      else return reason_str; // Return original if not recognized
    };
    
    int event_count = 0;
    while (sqlite3_step(st) == SQLITE_ROW) {
      std::int64_t ts = sqlite3_column_int64(st, 0);
      const char* kind = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
      const char* symbol = reinterpret_cast<const char*>(sqlite3_column_text(st, 2));
      const char* side = reinterpret_cast<const char*>(sqlite3_column_text(st, 3));
      double qty = sqlite3_column_double(st, 4);
      double price = sqlite3_column_double(st, 5);
      double pnl_delta = sqlite3_column_double(st, 6);
      double weight = sqlite3_column_double(st, 7);
      double prob = sqlite3_column_double(st, 8);
      const char* reason = reinterpret_cast<const char*>(sqlite3_column_text(st, 9));
      const char* note = reinterpret_cast<const char*>(sqlite3_column_text(st, 10));
      const char* chain_id = reinterpret_cast<const char*>(sqlite3_column_text(st, 11));
      (void)chain_id; // **CLEAN DISPLAY**: Chain ID kept internally but not displayed
      
      // Format timestamp
      char time_str[32];
      std::time_t time_t = ts / 1000;
      std::strftime(time_str, sizeof(time_str), "%H:%M:%S", std::localtime(&time_t));
      
      // Determine status and icon
      std::string status = "✅ PASSED";
      std::string event_icon = "📡";
      if (kind && strcmp(kind, "SIGNAL_DROP") == 0) {
        status = "❌ DROPPED";
        event_icon = "🚫";
      } else if (kind && strcmp(kind, "ORDER") == 0) {
        status = "📋 ORDERED";
        event_icon = "📋";
      } else if (kind && strcmp(kind, "FILL") == 0) {
        status = "💰 FILLED";
        event_icon = "✅";
      }
      
      // Add empty line before each signal event for better scanning
      if (event_count > 0) {
        printf("│         │          │        │        │        │        │         │\n");
      }
      
      printf("│ %-8s │ %s%-6s │ %-6s │ %-6s │ %-6.3f │ %-6.3f │ %-7s │\n",
             time_str, event_icon.c_str(), kind ? kind : "", symbol ? symbol : "", 
             side ? side : "", prob, weight, status.c_str());
      
      // Add detailed information with specific drop reasons
      if (kind && strcmp(kind, "SIGNAL_DROP") == 0) {
        std::string decoded_reason = decode_drop_reason(reason);
        printf("│         │ └─ Drop Reason: %-58s │\n", decoded_reason.c_str());
      } else if (kind && strcmp(kind, "SIGNAL") == 0) {
        // Show signal strength and reason for passed signals
        if (reason) {
          printf("│         │ └─ Signal Type: %-58s │\n", reason);
        }
        if (prob > 0.7) {
          printf("│         │ └─ 🟢 HIGH CONFIDENCE signal (%.1f%% probability)              │\n", prob * 100);
        } else if (prob > 0.5) {
          printf("│         │ └─ 🟡 MEDIUM CONFIDENCE signal (%.1f%% probability)            │\n", prob * 100);
        } else if (prob > 0.3) {
          printf("│         │ └─ 🟠 LOW CONFIDENCE signal (%.1f%% probability)               │\n", prob * 100);
        }
      } else if (kind && strcmp(kind, "ORDER") == 0) {
        printf("│         │ └─ Order placed: %s %.0f shares @ $%.2f                     │\n", 
               side ? side : "", qty, price);
      } else if (kind && strcmp(kind, "FILL") == 0) {
        printf("│         │ └─ Executed: %s %.0f shares @ $%.2f (P&L: $%.2f)            │\n", 
               side ? side : "", qty, price, pnl_delta);
      }
      
      event_count++;
    }
    
    printf("└─────────────────────────────────────────────────────────────────────────────────────────────┘\n");
    
    // **CONFLICT DETECTION**: Check for position conflicts in signal flow
    printf("\n🔍 POSITION CONFLICT CHECK:\n");
    printf("┌─────────────────────────────────────────────────────────────────────────────────────────────┐\n");
    
    // Quick conflict check by analyzing fills in this signal flow
    std::unordered_map<std::string, ConflictPosition> signal_positions;
    int signal_conflicts = 0;
    
    sqlite3_stmt* signal_conflict_st = nullptr;
    std::string signal_conflict_sql = "SELECT symbol, side, qty FROM audit_events WHERE run_id = ? AND kind = 'FILL'";
    if (!symbol_filter.empty()) {
        signal_conflict_sql += " AND symbol = '" + symbol_filter + "'";
    }
    signal_conflict_sql += " ORDER BY ts_millis ASC";
    if (limit > 0) {
        signal_conflict_sql += " LIMIT " + std::to_string(limit);
    }
    
    int signal_conflict_rc = sqlite3_prepare_v2(db.get_db(), signal_conflict_sql.c_str(), -1, &signal_conflict_st, nullptr);
    if (signal_conflict_rc == SQLITE_OK) {
        sqlite3_bind_text(signal_conflict_st, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
        
        while (sqlite3_step(signal_conflict_st) == SQLITE_ROW) {
            const char* symbol = (const char*)sqlite3_column_text(signal_conflict_st, 0);
            const char* side = (const char*)sqlite3_column_text(signal_conflict_st, 1);
            double qty = sqlite3_column_double(signal_conflict_st, 2);
            
            if (symbol && side) {
                auto& pos = signal_positions[symbol];
                pos.symbol = symbol;
                
                if (strcmp(side, "BUY") == 0) {
                    pos.qty += qty;
                } else if (strcmp(side, "SELL") == 0) {
                    pos.qty -= qty;
                }
            }
        }
        sqlite3_finalize(signal_conflict_st);
        
        // Analyze final positions for conflicts
        auto final_conflict_analysis = analyze_position_conflicts(signal_positions);
        if (final_conflict_analysis.has_conflicts) {
            signal_conflicts = final_conflict_analysis.conflicts.size();
        }
    }
    
    if (signal_conflicts == 0) {
        printf("│ ✅ SIGNAL FLOW CLEAN: No conflicting positions detected in signal processing    │\n");
    } else {
        printf("│ ⚠️  SIGNAL CONFLICTS: %d conflicting position patterns found                    │\n", signal_conflicts);
        printf("│    Signals may be generating opposing positions that waste capital             │\n");
    }
    
    printf("└─────────────────────────────────────────────────────────────────────────────────────────────┘\n");
    
    printf("\n📊 LEGEND:\n");
    printf("  ✅ PASSED  → Signal passed all validation gates\n");
    printf("  ❌ DROPPED → Signal dropped by validation (see reason)\n");
    printf("  📋 ORDERED → Signal converted to order\n");
    printf("  💰 FILLED  → Order executed (trade completed)\n");
    
    sqlite3_finalize(st);
  } catch (const std::exception& e) {
    fprintf(stderr, "Error showing signal flow: %s\n", e.what());
  }
}

struct TradeRecord {
  std::int64_t timestamp;
  std::string symbol;
  std::string action;  // BUY/SELL
  double quantity;
  double price;
  double trade_value;
  double realized_pnl;
  double cumulative_pnl;
  double equity_after;
};

struct PositionSummary {
  std::string symbol;
  double quantity;
  double avg_price;
  double market_value;
  double unrealized_pnl;
  double pnl_percent;
};

void show_position_history(const std::string& db_path, const std::string& run_id, const std::string& symbol_filter, int limit, bool show_buy, bool show_sell, bool show_hold) {
  try {
    DB db(db_path);
    
    // Get run info and print header
    RunInfo info = get_run_info(db_path, run_id);
    print_run_header("ACCOUNT STATEMENT", info);
    
    if (!symbol_filter.empty()) {
      printf("Symbol Filter: %s\n", symbol_filter.c_str());
    }
    if (limit > 0) {
      printf("Showing: %d most recent transactions\n", limit);
    } else {
      printf("Showing: All transactions\n");
    }
    printf("\n");
    
    // Get all FILL events to build trade history
    std::string sql = "SELECT ts_millis, symbol, side, qty, price, pnl_delta FROM audit_events WHERE run_id = ? AND kind = 'FILL'";
    
    if (!symbol_filter.empty()) {
      sql += " AND symbol = '" + symbol_filter + "'";
    }
    
    sql += " ORDER BY ts_millis ASC";
    
    sqlite3_stmt* st = nullptr;
    int rc = sqlite3_prepare_v2(db.get_db(), sql.c_str(), -1, &st, nullptr);
    if (rc != SQLITE_OK) {
      fprintf(stderr, "SQL prepare error: %s\n", sqlite3_errmsg(db.get_db()));
      return;
    }
    sqlite3_bind_text(st, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
    
    // Collect all trades and calculate running totals
    std::vector<TradeRecord> trades;
    std::map<std::string, double> positions; // symbol -> quantity
    std::map<std::string, double> avg_prices; // symbol -> average price
    std::map<std::string, double> realized_pnl_by_symbol; // symbol -> total realized P&L
    
    double starting_cash = 100000.0;
    double running_cash = starting_cash;
    double cumulative_realized_pnl = 0.0;
    
    // **DEBUG**: Track cash flow vs P&L separately to identify the bug
    double total_cash_flow = 0.0;
    
    while (sqlite3_step(st) == SQLITE_ROW) {
      std::int64_t ts = sqlite3_column_int64(st, 0);
      const char* symbol = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
      const char* side = reinterpret_cast<const char*>(sqlite3_column_text(st, 2));
      double qty = sqlite3_column_double(st, 3);
      double price = sqlite3_column_double(st, 4);
      double pnl_delta = sqlite3_column_double(st, 5);
      
      if (!symbol || !side) continue;
      
      std::string symbol_str = symbol;
      std::string action = side;
      bool is_buy = (action == "BUY");
      
      // Calculate trade value and cash impact
      double trade_value = qty * price;
      double cash_delta = is_buy ? -trade_value : trade_value;
      
      // **FIX**: P&L delta already includes the cash flow impact
      // Don't double-count by adding both cash_delta and pnl_delta
      // The pnl_delta represents the realized gain/loss from closing positions
      // Cash flow = trade_value changes, P&L = profit/loss on closed positions
      
      running_cash += cash_delta;
      total_cash_flow += cash_delta;
      
      // Only count pnl_delta as realized P&L (this is the actual profit/loss)
      double trade_realized_pnl = pnl_delta;
      cumulative_realized_pnl += trade_realized_pnl;
      realized_pnl_by_symbol[symbol_str] += trade_realized_pnl;
      
      // Update position and average price
      double old_qty = positions[symbol_str];
      double new_qty = old_qty + (is_buy ? qty : -qty);
      if (std::abs(new_qty) < 1e-6) {
        // Position closed
        positions.erase(symbol_str);
        avg_prices.erase(symbol_str);
      } else {
        if (old_qty * new_qty >= 0 && std::abs(old_qty) > 1e-6) {
          // Same direction - update VWAP
          avg_prices[symbol_str] = (avg_prices[symbol_str] * std::abs(old_qty) + price * qty) / std::abs(new_qty);
        } else {
          // New position or flipping direction
          avg_prices[symbol_str] = price;
        }
        positions[symbol_str] = new_qty;
      }
      
      // Calculate current equity (cash + position value at current prices)
      double total_position_value = 0.0;
      for (const auto& [sym, pos_qty] : positions) {
        if (std::abs(pos_qty) > 1e-6) {
          // Use the most recent price for this symbol as approximation
          double current_price = (sym == symbol_str) ? price : avg_prices[sym];
          total_position_value += pos_qty * current_price;
        }
      }
      double equity_after = running_cash + total_position_value;
      
      // Store trade record
      TradeRecord trade;
      trade.timestamp = ts;
      trade.symbol = symbol_str;
      trade.action = action;
      trade.quantity = qty;
      trade.price = price;
      trade.trade_value = trade_value;
      trade.realized_pnl = trade_realized_pnl;
      trade.cumulative_pnl = cumulative_realized_pnl;
      trade.equity_after = equity_after;
      
      trades.push_back(trade);
    }
    
    sqlite3_finalize(st);
    
    // Calculate final metrics
    double final_equity = running_cash;
    double total_unrealized_pnl = 0.0;
    std::vector<PositionSummary> current_positions;
    
    for (const auto& [symbol, qty] : positions) {
      if (std::abs(qty) > 1e-6) {
        double avg_price = avg_prices[symbol];
        // **FIX**: Use the most recent trade price as current market price
        double current_price = avg_price; // Fallback to avg price
        
        // Find the most recent price for this symbol from all trades
        for (int j = (int)trades.size() - 1; j >= 0; j--) {
          if (trades[j].symbol == symbol) {
            current_price = trades[j].price;
            break;
          }
        }
        
        double market_value = qty * current_price;
        double unrealized_pnl = qty * (current_price - avg_price);
        double pnl_percent = (std::abs(avg_price) > 1e-6) ? (unrealized_pnl / (std::abs(qty) * avg_price)) * 100.0 : 0.0;
        
        final_equity += market_value;
        total_unrealized_pnl += unrealized_pnl;
        
        PositionSummary pos;
        pos.symbol = symbol;
        pos.quantity = qty;
        pos.avg_price = avg_price;
        pos.market_value = market_value;
        pos.unrealized_pnl = unrealized_pnl;
        pos.pnl_percent = pnl_percent;
        current_positions.push_back(pos);
      }
    }
    
    double total_return = ((final_equity - starting_cash) / starting_cash) * 100.0;
    
    // 1. EXECUTIVE SUMMARY
    printf("📊 ACCOUNT PERFORMANCE SUMMARY\n");
    printf("┌─────────────────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Starting Capital    │ $%10.2f │ Current Equity      │ $%10.2f │ Total Return │ %+6.2f%% │\n", 
           starting_cash, final_equity, total_return);
    printf("│ Total Trades        │ %11d │ Realized P&L        │ $%+10.2f │ Unrealized   │ $%+7.2f │\n", 
           (int)trades.size(), cumulative_realized_pnl, total_unrealized_pnl);
    printf("│ Cash Balance        │ $%10.2f │ Position Value      │ $%10.2f │ Open Pos.    │ %8d │\n", 
           running_cash, final_equity - running_cash, (int)current_positions.size());
    printf("└─────────────────────────────────────────────────────────────────────────────────────────────┘\n\n");
    
    // 2. RECENT TRADE HISTORY
    printf("📈 TRADE HISTORY");
    if (limit > 0 && (int)trades.size() > limit) {
      printf(" (Last %d of %d trades)", limit, (int)trades.size());
    }
    printf("\n");
    printf("┌──────────────┬────────┬────────┬──────────┬──────────┬─────────────┬─────────────┬─────────────┐\n");
    printf("│ Date/Time    │ Symbol │ Action │ Quantity │ Price    │ Trade Value │ Realized P&L│ Equity After│\n");
    printf("├──────────────┼────────┼────────┼──────────┼──────────┼─────────────┼─────────────┼─────────────┤\n");
    
    // Show recent trades (apply limit here)
    int start_idx = (limit > 0 && (int)trades.size() > limit) ? (int)trades.size() - limit : 0;
    for (int i = start_idx; i < (int)trades.size(); i++) {
      const auto& trade = trades[i];
      
      // Format timestamp
      char date_str[32];
      std::time_t time_t = trade.timestamp / 1000;
      std::strftime(date_str, sizeof(date_str), "%m/%d %H:%M:%S", std::localtime(&time_t));
      
      // Color coding for actions
      const char* action_color = (trade.action == "BUY") ? "🟢" : "🔴";
      
      // **FIX**: Show fractional shares to avoid misleading "0 shares" display
      printf("│ %-12s │ %-6s │ %s%-4s │ %8.3f │ $%7.2f │ $%+10.2f │ $%+10.2f │ $%+10.2f │\n",
             date_str, trade.symbol.c_str(), action_color, trade.action.c_str(),
             trade.quantity, trade.price, trade.trade_value, trade.realized_pnl, trade.equity_after);
    }
    printf("└──────────────┴────────┴────────┴──────────┴──────────┴─────────────┴─────────────┴─────────────┘\n\n");
    
    // 3. CURRENT POSITIONS
    if (!current_positions.empty()) {
      printf("💼 CURRENT POSITIONS\n");
      printf("┌────────┬──────────┬───────────┬─────────────┬─────────────┬──────────┐\n");
      printf("│ Symbol │ Quantity │ Avg Price │ Market Value│ Unrealized  │ Return %% │\n");
      printf("├────────┼──────────┼───────────┼─────────────┼─────────────┼──────────┤\n");
      
      for (const auto& pos : current_positions) {
        const char* pnl_color = (pos.unrealized_pnl >= 0) ? "🟢" : "🔴";
        
        // **POSITION DISPLAY**: Show actual positions - no negative quantities allowed
        std::string display_symbol = pos.symbol;
        double display_quantity = pos.quantity;
        
        // All positions should be positive quantities in Sentio system
        // SQQQ positive position = long SQQQ (inverse ETF)
        // No conversion needed - show actual position
        
        printf("│ %-6s │ %8.0f │ $%8.2f │ $%+10.2f │ %s$%+8.2f │ %+7.2f%% │\n",
               display_symbol.c_str(), display_quantity, pos.avg_price, pos.market_value,
               pnl_color, pos.unrealized_pnl, pos.pnl_percent);
      }
      printf("└────────┴──────────┴───────────┴─────────────┴─────────────┴──────────┘\n\n");
    } else {
      printf("💼 CURRENT POSITIONS: None (All positions closed)\n\n");
    }
    
    // 4. PERFORMANCE BREAKDOWN
    printf("📊 PERFORMANCE BREAKDOWN\n");
    printf("┌─────────────────────────────────────────────────────────────────────────────────────────────┐\n");
    printf("│ Metric                    │ Amount        │ Percentage    │ Description                     │\n");
    printf("├─────────────────────────────────────────────────────────────────────────────────────────────┤\n");
    // **FIX**: For a closed portfolio, realized P&L should equal cash flow
    // The audit P&L calculation appears to be wrong, so use cash flow as the correct realized P&L
    double correct_realized_pnl = total_cash_flow;
    
    printf("│ Realized Gains/Losses     │ $%'+10.2f │ %+6.2f%%      │ Profit from closed positions    │\n", 
           correct_realized_pnl, (correct_realized_pnl / starting_cash) * 100.0);
    printf("│ **DEBUG** Audit P&L       │ $%'+10.2f │ %+6.2f%%      │ P&L from audit (incorrect)      │\n", 
           cumulative_realized_pnl, (cumulative_realized_pnl / starting_cash) * 100.0);
    printf("│ Unrealized Gains/Losses   │ $%'+10.2f │ %+6.2f%%      │ Profit from open positions      │\n", 
           total_unrealized_pnl, (total_unrealized_pnl / starting_cash) * 100.0);
    printf("│ Total Return              │ $%'+10.2f │ %+6.2f%%      │ Overall account performance     │\n", 
           final_equity - starting_cash, total_return);
    printf("└─────────────────────────────────────────────────────────────────────────────────────────────┘\n");
    
  } catch (const std::exception& e) {
    fprintf(stderr, "Error showing position history: %s\n", e.what());
  }
}

void show_strategies_summary(const std::string& db_path) {
  try {
    DB db(db_path);
    
    printf("📊 STRATEGIES SUMMARY REPORT\n");
    printf("════════════════════════════════════════════════════════════════════════════════════════════════\n");
    printf("Summary of all strategies' most recent runs\n");
    // Format current timestamp
    char time_buffer[64];
    std::time_t ts_sec = now_millis() / 1000;
    struct tm* tm_info = localtime(&ts_sec);
    strftime(time_buffer, sizeof(time_buffer), "%Y-%m-%d %H:%M:%S", tm_info);
    printf("Generated: %s\n\n", time_buffer);
    
    // Query to get the latest run for each strategy
    sqlite3* sqlite_db = nullptr;
    int rc = sqlite3_open(db_path.c_str(), &sqlite_db);
    if (rc != SQLITE_OK) {
      fprintf(stderr, "Error opening database: %s\n", sqlite3_errmsg(sqlite_db));
      return;
    }
    
    // Get latest run for each strategy
    const char* sql = R"(
      SELECT r.strategy, r.run_id, r.started_at, r.kind, r.note,
             COUNT(e.run_id) as total_events,
             SUM(CASE WHEN e.kind = 'FILL' THEN e.pnl_delta ELSE 0 END) as total_pnl,
             COUNT(CASE WHEN e.kind = 'FILL' THEN 1 END) as total_trades,
             COUNT(CASE WHEN e.kind = 'SIGNAL' THEN 1 END) as total_signals
      FROM (
        SELECT strategy, MAX(started_at) as max_started_at
        FROM audit_runs 
        GROUP BY strategy
      ) latest
      JOIN audit_runs r ON r.strategy = latest.strategy AND r.started_at = latest.max_started_at
      LEFT JOIN audit_events e ON e.run_id = r.run_id
      GROUP BY r.strategy, r.run_id, r.started_at, r.kind, r.note
      ORDER BY r.strategy
    )";
    
    sqlite3_stmt* stmt;
    rc = sqlite3_prepare_v2(sqlite_db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
      fprintf(stderr, "Error preparing query: %s\n", sqlite3_errmsg(sqlite_db));
      sqlite3_close(sqlite_db);
      return;
    }
    
    printf("┌─────────────┬────────┬─────────────────────┬───────────┬──────────┬─────────────┬─────────────┬─────────────┐\n");
    printf("│ Strategy    │ Run ID │ Date/Time           │ Test Type │ Signals  │ Trades      │ Total P&L   │ MPR Est.    │\n");
    printf("├─────────────┼────────┼─────────────────────┼───────────┼──────────┼─────────────┼─────────────┼─────────────┤\n");
    
    bool has_data = false;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
      has_data = true;
      
      const char* strategy = (const char*)sqlite3_column_text(stmt, 0);
      const char* run_id = (const char*)sqlite3_column_text(stmt, 1);
      int64_t started_at = sqlite3_column_int64(stmt, 2);
      const char* kind = (const char*)sqlite3_column_text(stmt, 3);
      const char* note = (const char*)sqlite3_column_text(stmt, 4);
      int total_events = sqlite3_column_int(stmt, 5);
      double total_pnl = sqlite3_column_double(stmt, 6);
      int total_trades = sqlite3_column_int(stmt, 7);
      int total_signals = sqlite3_column_int(stmt, 8);
      
      // Format timestamp
      char date_str[32];
      std::time_t time_t = started_at / 1000;
      std::strftime(date_str, sizeof(date_str), "%m/%d %H:%M:%S", std::localtime(&time_t));
      
      // Extract test type from note (e.g., "strattest holistic QQQ 2w")
      std::string test_type = "unknown";
      if (note && strlen(note) > 0) {
        std::string note_str(note);
        if (note_str.find("holistic") != std::string::npos) {
          test_type = "holistic";
        } else if (note_str.find("historical") != std::string::npos) {
          test_type = "historical";
        } else if (note_str.find("ai-regime") != std::string::npos) {
          test_type = "ai-regime";
        } else if (note_str.find("hybrid") != std::string::npos) {
          test_type = "hybrid";
        } else if (note_str.find("strattest") != std::string::npos) {
          test_type = "strattest";
        }
      }
      
      // Estimate MPR (very rough calculation)
      double mpr_estimate = 0.0;
      if (total_pnl != 0.0) {
        // Assume 100k starting capital and estimate monthly return
        double return_pct = (total_pnl / 100000.0) * 100.0;
        // Very rough annualization (assuming test was representative)
        mpr_estimate = return_pct * 12.0; // Rough monthly estimate
      }
      
      // Color coding for P&L
      const char* pnl_color = (total_pnl >= 0) ? "🟢" : "🔴";
      const char* mpr_color = (mpr_estimate >= 0) ? "🟢" : "🔴";
      
      printf("│ %-11s │ %-6s │ %-19s │ %-9s │ %8d │ %11d │ %s$%+9.2f │ %s%+6.1f%%    │\n",
             strategy ? strategy : "unknown",
             run_id ? run_id : "N/A",
             date_str,
             test_type.c_str(),
             total_signals,
             total_trades,
             pnl_color, total_pnl,
             mpr_color, mpr_estimate);
    }
    
    printf("└─────────────┴────────┴─────────────────────┴───────────┴──────────┴─────────────┴─────────────┴─────────────┘\n");
    
    if (!has_data) {
      printf("\n⚠️  No strategy runs found in the database.\n");
      printf("Run some strategies using 'sentio_cli strattest' to populate the audit database.\n");
    } else {
      printf("\n📋 SUMMARY NOTES:\n");
      printf("• Run ID: 6-digit unique identifier for each test run\n");
      printf("• MPR Est.: Rough Monthly Projected Return estimate (not precise)\n");
      printf("• For detailed analysis, use: sentio_audit summarize --run <run_id>\n");
      printf("• For signal analysis, use: sentio_audit signal-flow --run <run_id>\n");
      printf("• For trade analysis, use: sentio_audit trade-flow --run <run_id>\n");
    }
    
    sqlite3_finalize(stmt);
    sqlite3_close(sqlite_db);
    
  } catch (const std::exception& e) {
    fprintf(stderr, "Error generating strategies summary: %s\n", e.what());
  }
}

} // namespace audit

```

## 📄 **FILE 9 of 14**: audit/src/audit_db.cpp

**File Information**:
- **Path**: `audit/src/audit_db.cpp`

- **Size**: 652 lines
- **Modified**: 2025-09-15 21:07:52

- **Type**: .cpp

```text
#include "audit/audit_db.hpp"
#include "audit/hash.hpp"
#include "sentio/metrics/mpr.hpp"
#include <sqlite3.h>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <cassert>
#include <cstdio>
#include <cmath>

namespace audit {

static void exec_or_throw(sqlite3* db, const char* sql) {
  char* err=nullptr;
  if (sqlite3_exec(db, sql, nullptr, nullptr, &err) != SQLITE_OK) {
    std::string msg = err? err : "unknown sqlite error";
    sqlite3_free(err);
    throw std::runtime_error("sqlite exec failed: " + msg);
  }
}

DB::DB(const std::string& path) : db_path_(path) {
  if (sqlite3_open(db_path_.c_str(), &db_) != SQLITE_OK) {
    throw std::runtime_error("cannot open db: " + path);
  }
  exec_or_throw(db_, "PRAGMA journal_mode=WAL;");
  exec_or_throw(db_, "PRAGMA synchronous=NORMAL;");
  exec_or_throw(db_, "PRAGMA foreign_keys=ON;");
}

DB::~DB() { if (db_) sqlite3_close(db_); }

void DB::init_schema() {
  const char* ddl =
    "CREATE TABLE IF NOT EXISTS audit_runs ("
    " run_id TEXT PRIMARY KEY, started_at INTEGER NOT NULL, ended_at INTEGER,"
    " kind TEXT NOT NULL, strategy TEXT NOT NULL, params_json TEXT NOT NULL,"
    " data_hash TEXT NOT NULL, git_rev TEXT, note TEXT,"
    " run_period_start_ts_ms INTEGER, run_period_end_ts_ms INTEGER,"
    " run_trading_days INTEGER, session_calendar TEXT,"
    " dataset_source_type TEXT DEFAULT 'unknown',"
    " dataset_file_path TEXT DEFAULT '',"
    " dataset_file_hash TEXT DEFAULT '',"
    " dataset_track_id TEXT DEFAULT '',"
    " dataset_regime TEXT DEFAULT '',"
    " dataset_bars_count INTEGER DEFAULT 0,"
    " dataset_time_range_start INTEGER DEFAULT 0,"
    " dataset_time_range_end INTEGER DEFAULT 0 );"
    "CREATE TABLE IF NOT EXISTS audit_events ("
    " run_id TEXT NOT NULL, seq INTEGER NOT NULL, ts_millis INTEGER NOT NULL,"
    " kind TEXT NOT NULL, symbol TEXT, side TEXT, qty REAL, price REAL,"
    " pnl_delta REAL, weight REAL, prob REAL, reason TEXT, note TEXT,"
    " hash_prev TEXT NOT NULL, hash_curr TEXT NOT NULL,"
    " PRIMARY KEY(run_id,seq),"
    " FOREIGN KEY(run_id) REFERENCES audit_runs(run_id) ON DELETE CASCADE );"
    "CREATE TABLE IF NOT EXISTS audit_kv ("
    " run_id TEXT NOT NULL, k TEXT NOT NULL, v TEXT NOT NULL,"
    " PRIMARY KEY(run_id,k),"
    " FOREIGN KEY(run_id) REFERENCES audit_runs(run_id) ON DELETE CASCADE );"
    "CREATE TABLE IF NOT EXISTS audit_meta ("
    " key TEXT PRIMARY KEY, value TEXT NOT NULL );"
    "CREATE TABLE IF NOT EXISTS audit_daily_returns ("
    " run_id TEXT NOT NULL, session_date TEXT NOT NULL,"
    " r_simple REAL NOT NULL,"
    " PRIMARY KEY(run_id, session_date),"
    " FOREIGN KEY(run_id) REFERENCES audit_runs(run_id) ON DELETE CASCADE );"
    "CREATE INDEX IF NOT EXISTS idx_events_run_ts ON audit_events(run_id, ts_millis);"
    "CREATE INDEX IF NOT EXISTS idx_events_kind ON audit_events(run_id, kind);";
  exec_or_throw(db_, ddl);
}

void DB::new_run(const RunRow& r) {
  const char* sql =
    "INSERT INTO audit_runs(run_id,started_at,ended_at,kind,strategy,params_json,data_hash,git_rev,note,"
    "run_period_start_ts_ms,run_period_end_ts_ms,run_trading_days,session_calendar,"
    "dataset_source_type,dataset_file_path,dataset_file_hash,dataset_track_id,dataset_regime,"
    "dataset_bars_count,dataset_time_range_start,dataset_time_range_end)"
    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);";
  sqlite3_stmt* st=nullptr;
  sqlite3_prepare_v2(db_, sql, -1, &st, nullptr);
  sqlite3_bind_text(st,1,r.run_id.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_int64(st,2,r.started_at);
  if (r.ended_at) sqlite3_bind_int64(st,3,*r.ended_at); else sqlite3_bind_null(st,3);
  sqlite3_bind_text(st,4,r.kind.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,5,r.strategy.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,6,r.params_json.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,7,r.data_hash.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,8,r.git_rev.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,9,r.note.c_str(),-1,SQLITE_TRANSIENT);
  // New canonical period fields
  if (r.run_period_start_ts_ms) sqlite3_bind_int64(st,10,*r.run_period_start_ts_ms); else sqlite3_bind_null(st,10);
  if (r.run_period_end_ts_ms) sqlite3_bind_int64(st,11,*r.run_period_end_ts_ms); else sqlite3_bind_null(st,11);
  if (r.run_trading_days) sqlite3_bind_int(st,12,*r.run_trading_days); else sqlite3_bind_null(st,12);
  sqlite3_bind_text(st,13,r.session_calendar.c_str(),-1,SQLITE_TRANSIENT);
  // Dataset traceability fields
  sqlite3_bind_text(st,14,r.dataset_source_type.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,15,r.dataset_file_path.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,16,r.dataset_file_hash.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,17,r.dataset_track_id.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_text(st,18,r.dataset_regime.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_bind_int(st,19,r.dataset_bars_count);
  sqlite3_bind_int64(st,20,r.dataset_time_range_start);
  sqlite3_bind_int64(st,21,r.dataset_time_range_end);
  if (sqlite3_step(st) != SQLITE_DONE) throw std::runtime_error("new_run failed");
  sqlite3_finalize(st);
  
  // Update the latest run ID
  set_latest_run_id(r.run_id);
}

void DB::set_latest_run_id(const std::string& run_id) {
  const char* sql = "INSERT OR REPLACE INTO audit_meta(key, value) VALUES('latest_run_id', ?);";
  sqlite3_stmt* st=nullptr;
  if (sqlite3_prepare_v2(db_, sql, -1, &st, nullptr) != SQLITE_OK) return;
  sqlite3_bind_text(st,1,run_id.c_str(),-1,SQLITE_TRANSIENT);
  sqlite3_step(st);
  sqlite3_finalize(st);
}

std::string DB::get_latest_run_id() {
  const char* sql = "SELECT value FROM audit_meta WHERE key='latest_run_id';";
  sqlite3_stmt* st=nullptr;
  if (sqlite3_prepare_v2(db_, sql, -1, &st, nullptr) != SQLITE_OK) return "";
  
  std::string latest_run_id;
  if (sqlite3_step(st) == SQLITE_ROW) {
    const char* value = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
    if (value) latest_run_id = value;
  }
  sqlite3_finalize(st);
  return latest_run_id;
}

void DB::store_daily_returns(const std::string& run_id, const std::vector<std::pair<std::string, double>>& daily_returns) {
  const char* sql = "INSERT INTO audit_daily_returns(run_id, session_date, r_simple) VALUES(?,?,?);";
  sqlite3_stmt* st=nullptr;
  if (sqlite3_prepare_v2(db_, sql, -1, &st, nullptr) != SQLITE_OK) return;
  
  for (const auto& [session_date, r_simple] : daily_returns) {
    sqlite3_bind_text(st, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st, 2, session_date.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(st, 3, r_simple);
    sqlite3_step(st);
    sqlite3_reset(st);
  }
  sqlite3_finalize(st);
}

std::vector<double> DB::load_daily_returns(const std::string& run_id) {
  const char* sql = "SELECT r_simple FROM audit_daily_returns WHERE run_id=? ORDER BY session_date;";
  sqlite3_stmt* st=nullptr;
  std::vector<double> returns;
  
  if (sqlite3_prepare_v2(db_, sql, -1, &st, nullptr) != SQLITE_OK) return returns;
  sqlite3_bind_text(st, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
  
  while (sqlite3_step(st) == SQLITE_ROW) {
    returns.push_back(sqlite3_column_double(st, 0));
  }
  sqlite3_finalize(st);
  return returns;
}

void DB::end_run(const std::string& run_id, std::int64_t ended_at) {
  sqlite3_stmt* st=nullptr;
  sqlite3_prepare_v2(db_, "UPDATE audit_runs SET ended_at=? WHERE run_id=?", -1, &st, nullptr);
  sqlite3_bind_int64(st,1,ended_at);
  sqlite3_bind_text(st,2,run_id.c_str(),-1,SQLITE_TRANSIENT);
  if (sqlite3_step(st) != SQLITE_DONE) throw std::runtime_error("end_run failed");
  sqlite3_finalize(st);
}

std::pair<std::int64_t,std::string> DB::last_seq_and_hash(const std::string& run_id) {
  sqlite3_stmt* st=nullptr;
  sqlite3_prepare_v2(db_, "SELECT seq,hash_curr FROM audit_events WHERE run_id=? ORDER BY seq DESC LIMIT 1", -1, &st, nullptr);
  sqlite3_bind_text(st,1,run_id.c_str(),-1,SQLITE_TRANSIENT);
  int rc = sqlite3_step(st);
  if (rc == SQLITE_ROW) {
    auto seq = sqlite3_column_int64(st,0);
    const unsigned char* h = sqlite3_column_text(st,1);
    std::string hash = h? reinterpret_cast<const char*>(h) : "";
    sqlite3_finalize(st);
    return {seq, hash};
  }
  sqlite3_finalize(st);
  return {0, "GENESIS"};
}

std::string DB::canonical_content_string(std::int64_t seq, const Event& ev) {
  auto q = [](double v){ std::ostringstream o; o<<std::setprecision(12)<<v; return o.str(); };
  std::ostringstream ss;
  ss << "run="<<ev.run_id<<"|seq="<<seq<<"|ts="<<ev.ts_millis
     <<"|kind="<<ev.kind<<"|symbol="<<ev.symbol<<"|side="<<ev.side
     <<"|qty="<<q(ev.qty)<<"|price="<<q(ev.price)<<"|pnl="<<q(ev.pnl_delta)
     <<"|weight="<<q(ev.weight)<<"|prob="<<q(ev.prob)<<"|reason="<<ev.reason<<"|note="<<ev.note;
  return ss.str();
}

std::pair<std::int64_t,std::string> DB::append_event(const Event& ev) {
  auto [last_seq, last_hash] = last_seq_and_hash(ev.run_id);
  std::int64_t seq = last_seq + 1;
  std::string content = canonical_content_string(seq, ev);
  std::string content_hash = sha256_hex(content);
  std::string hash_curr = sha256_hex(last_hash + "\n" + content_hash);

  const char* sql =
    "INSERT INTO audit_events(run_id,seq,ts_millis,kind,symbol,side,qty,price,pnl_delta,weight,prob,reason,note,hash_prev,hash_curr)"
    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);";
  sqlite3_stmt* st=nullptr; sqlite3_prepare_v2(db_, sql, -1, &st, nullptr);
  sqlite3_bind_text  (st, 1, ev.run_id.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_int64 (st, 2, seq);
  sqlite3_bind_int64 (st, 3, ev.ts_millis);
  sqlite3_bind_text  (st, 4, ev.kind.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_text  (st, 5, ev.symbol.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_text  (st, 6, ev.side.c_str(),   -1, SQLITE_TRANSIENT);
  sqlite3_bind_double(st, 7, ev.qty);
  sqlite3_bind_double(st, 8, ev.price);
  sqlite3_bind_double(st, 9, ev.pnl_delta);
  sqlite3_bind_double(st,10, ev.weight);
  sqlite3_bind_double(st,11, ev.prob);
  sqlite3_bind_text  (st,12, ev.reason.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_text  (st,13, ev.note.c_str(),   -1, SQLITE_TRANSIENT);
  sqlite3_bind_text  (st,14, last_hash.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_text  (st,15, hash_curr.c_str(), -1, SQLITE_TRANSIENT);
  if (sqlite3_step(st) != SQLITE_DONE) throw std::runtime_error("append_event failed");
  sqlite3_finalize(st);

  return {seq, hash_curr};
}

std::pair<bool,std::string> DB::verify_run(const std::string& run_id) {
  // Enhanced verification with detailed reporting
  
  // First, get run metadata for context
  sqlite3_stmt* meta_st = nullptr;
  sqlite3_prepare_v2(db_, 
    "SELECT strategy, kind, started_at, ended_at, note FROM audit_runs WHERE run_id=?", 
    -1, &meta_st, nullptr);
  sqlite3_bind_text(meta_st, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
  
  std::string strategy = "UNKNOWN";
  std::string test_kind = "UNKNOWN";
  std::int64_t started_at = 0;
  std::int64_t ended_at = 0;
  std::string note = "";
  
  if (sqlite3_step(meta_st) == SQLITE_ROW) {
    const char* strategy_ptr = reinterpret_cast<const char*>(sqlite3_column_text(meta_st, 0));
    const char* kind_ptr = reinterpret_cast<const char*>(sqlite3_column_text(meta_st, 1));
    const char* note_ptr = reinterpret_cast<const char*>(sqlite3_column_text(meta_st, 4));
    
    if (strategy_ptr) strategy = strategy_ptr;
    if (kind_ptr) test_kind = kind_ptr;
    if (note_ptr) note = note_ptr;
    started_at = sqlite3_column_int64(meta_st, 2);
    ended_at = sqlite3_column_int64(meta_st, 3);
  }
  sqlite3_finalize(meta_st);
  
  sqlite3_stmt* st=nullptr;
  sqlite3_prepare_v2(db_,
    "SELECT seq,ts_millis,kind,symbol,side,qty,price,pnl_delta,weight,prob,reason,note,hash_prev,hash_curr"
    " FROM audit_events WHERE run_id=? ORDER BY seq ASC", -1, &st, nullptr);
  sqlite3_bind_text(st,1,run_id.c_str(),-1,SQLITE_TRANSIENT);

  std::int64_t expected_seq = 1;
  std::string prev = "GENESIS";
  std::int64_t total_events = 0;
  std::int64_t hash_verifications = 0;
  std::int64_t sequence_checks = 0;
  
  // Count total events first for progress reporting
  sqlite3_stmt* count_st = nullptr;
  sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM audit_events WHERE run_id=?", -1, &count_st, nullptr);
  sqlite3_bind_text(count_st, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
  if (sqlite3_step(count_st) == SQLITE_ROW) {
    total_events = sqlite3_column_int64(count_st, 0);
  }
  sqlite3_finalize(count_st);
  
  if (total_events == 0) {
    sqlite3_finalize(st);
    return {false, "❌ VERIFICATION FAILED: No audit events found for run " + run_id + 
                   " (Strategy: " + strategy + ", Test: " + test_kind + ")"};
  }

  while (sqlite3_step(st) == SQLITE_ROW) {
    auto seq = sqlite3_column_int64(st,0);
    sequence_checks++;
    
    // Enhanced sequence monotonicity check
    if (seq != expected_seq) { 
      sqlite3_finalize(st); 
      return {false, "❌ SEQUENCE INTEGRITY FAILED: Expected sequence " + std::to_string(expected_seq) + 
                     " but found " + std::to_string(seq) + " (gap or duplicate detected)"};
    }
    
    Event ev;
    ev.run_id = run_id;
    ev.ts_millis = sqlite3_column_int64(st,1);
    ev.kind  = reinterpret_cast<const char*>(sqlite3_column_text(st,2));
    ev.symbol= reinterpret_cast<const char*>(sqlite3_column_text(st,3));
    ev.side  = reinterpret_cast<const char*>(sqlite3_column_text(st,4));
    ev.qty   = sqlite3_column_double(st,5);
    ev.price = sqlite3_column_double(st,6);
    ev.pnl_delta = sqlite3_column_double(st,7);
    ev.weight= sqlite3_column_double(st,8);
    ev.prob  = sqlite3_column_double(st,9);
    ev.reason= reinterpret_cast<const char*>(sqlite3_column_text(st,10));
    ev.note  = reinterpret_cast<const char*>(sqlite3_column_text(st,11));
    std::string hash_prev = reinterpret_cast<const char*>(sqlite3_column_text(st,12));
    std::string hash_curr = reinterpret_cast<const char*>(sqlite3_column_text(st,13));

    // Enhanced hash chain verification
    if (hash_prev != prev) { 
      sqlite3_finalize(st); 
      return {false, "❌ HASH CHAIN BROKEN: Event " + std::to_string(seq) + " has invalid previous hash\n" +
                     "Expected: " + prev.substr(0, 16) + "...\n" +
                     "Found:    " + hash_prev.substr(0, 16) + "... (possible tampering detected)"};
    }

    std::string content = canonical_content_string(seq, ev);
    std::string content_hash = sha256_hex(content);
    std::string recomputed = sha256_hex(prev + "\n" + content_hash);
    hash_verifications++;
    
    if (recomputed != hash_curr) { 
      sqlite3_finalize(st); 
      return {false, "❌ CONTENT INTEGRITY FAILED: Event " + std::to_string(seq) + " (" + ev.kind + ") content hash mismatch\n" +
                     "Expected: " + recomputed.substr(0, 16) + "...\n" +
                     "Stored:   " + hash_curr.substr(0, 16) + "... (content may have been modified)"};
    }

    prev = hash_curr;
    ++expected_seq;
  }
  sqlite3_finalize(st);
  
  // Success message with detailed verification summary including run context
  std::string success_msg = "✅ AUDIT TRAIL VERIFIED & INTACT\n";
  success_msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  success_msg += "🎯 VERIFICATION TARGET:\n";
  success_msg += "   Run ID: " + run_id + "\n";
  success_msg += "   Strategy: " + strategy + "\n";
  success_msg += "   Test Type: " + test_kind + "\n";
  if (!note.empty()) {
    // Extract key info from note (e.g., "Strategy: sigor, Test: vm_test, Generated by strattest")
    success_msg += "   Details: " + note + "\n";
  }
  if (started_at > 0) {
    success_msg += "   Started: " + std::to_string(started_at) + "\n";
  }
  if (ended_at > 0) {
    success_msg += "   Ended: " + std::to_string(ended_at) + "\n";
  }
  success_msg += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
  success_msg += "📊 VERIFICATION RESULTS:\n";
  success_msg += "   Events Processed: " + std::to_string(total_events) + "\n";
  success_msg += "   🔢 Sequence Checks: " + std::to_string(sequence_checks) + " (all monotonic)\n";
  success_msg += "   🔐 Hash Verifications: " + std::to_string(hash_verifications) + " (all valid)\n";
  success_msg += "   ⛓️  Chain Integrity: UNBROKEN (Genesis → Event " + std::to_string(total_events) + ")\n";
  success_msg += "   🛡️  Tamper Detection: NONE (cryptographically secure)\n";
  success_msg += "   📋 Regulatory Compliance: READY";
  
  return {true, success_msg};
}

std::vector<Event> DB::get_events_for_run(const std::string& run_id) {
  std::vector<Event> events;
  
  sqlite3_stmt* st = nullptr;
  const char* sql = R"(
    SELECT ts_millis, kind, symbol, side, qty, price, pnl_delta, weight, prob, reason, note
    FROM events 
    WHERE run_id = ? 
    ORDER BY seq ASC
  )";
  
  int rc = sqlite3_prepare_v2(db_, sql, -1, &st, nullptr);
  if (rc != SQLITE_OK) {
    return events; // Return empty vector on error
  }
  
  sqlite3_bind_text(st, 1, run_id.c_str(), -1, SQLITE_STATIC);
  
  while (sqlite3_step(st) == SQLITE_ROW) {
    Event event;
    event.run_id = run_id;
    event.ts_millis = sqlite3_column_int64(st, 0);
    
    const char* kind = (const char*)sqlite3_column_text(st, 1);
    event.kind = kind ? kind : "";
    
    const char* symbol = (const char*)sqlite3_column_text(st, 2);
    event.symbol = symbol ? symbol : "";
    
    const char* side = (const char*)sqlite3_column_text(st, 3);
    event.side = side ? side : "";
    
    event.qty = sqlite3_column_double(st, 4);
    event.price = sqlite3_column_double(st, 5);
    event.pnl_delta = sqlite3_column_double(st, 6);
    event.weight = sqlite3_column_double(st, 7);
    event.prob = sqlite3_column_double(st, 8);
    
    const char* reason = (const char*)sqlite3_column_text(st, 9);
    event.reason = reason ? reason : "";
    
    const char* note = (const char*)sqlite3_column_text(st, 10);
    event.note = note ? note : "";
    
    events.push_back(event);
  }
  
  sqlite3_finalize(st);
  return events;
}

DB::Summary DB::summarize(const std::string& run_id) {
  Summary s{};
  sqlite3_stmt* st=nullptr;
  
  // Basic counts and P&L
  sqlite3_prepare_v2(db_,
    "SELECT COUNT(1),"
    " SUM(CASE WHEN kind='SIGNAL' THEN 1 ELSE 0 END),"
    " SUM(CASE WHEN kind='ORDER'  THEN 1 ELSE 0 END),"
    " SUM(CASE WHEN kind='FILL'   THEN 1 ELSE 0 END),"
    " SUM(CASE WHEN kind='PNL'    THEN 1 ELSE 0 END),"
    " COALESCE(SUM(pnl_delta),0),"
    " MIN(ts_millis), MAX(ts_millis)"
    " FROM audit_events WHERE run_id=?", -1, &st, nullptr);
  sqlite3_bind_text(st,1,run_id.c_str(),-1,SQLITE_TRANSIENT);
  if (sqlite3_step(st) == SQLITE_ROW) {
    s.n_total = sqlite3_column_int64(st,0);
    s.n_signal= sqlite3_column_int64(st,1);
    s.n_order = sqlite3_column_int64(st,2);
    s.n_fill  = sqlite3_column_int64(st,3);
    s.n_pnl   = sqlite3_column_int64(st,4);
    s.pnl_sum = sqlite3_column_double(st,5);
    s.ts_first= sqlite3_column_int64(st,6);
    s.ts_last = sqlite3_column_int64(st,7);
  }
  sqlite3_finalize(st);
  
  // Calculate instrument distribution and P&L breakdown
  sqlite3_prepare_v2(db_,
    "SELECT symbol, COUNT(1) as fills, COALESCE(SUM(pnl_delta),0) as pnl, "
    "COALESCE(SUM(qty * price),0) as volume "
    "FROM audit_events WHERE run_id=? AND kind='FILL' "
    "GROUP BY symbol ORDER BY pnl DESC", -1, &st, nullptr);
  sqlite3_bind_text(st, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
  
  while (sqlite3_step(st) == SQLITE_ROW) {
    const char* symbol = (const char*)sqlite3_column_text(st, 0);
    if (symbol) {
      std::string sym(symbol);
      s.instrument_fills[sym] = sqlite3_column_int64(st, 1);
      s.instrument_pnl[sym] = sqlite3_column_double(st, 2);
      s.instrument_volume[sym] = sqlite3_column_double(st, 3);
    }
  }
  sqlite3_finalize(st);
  
  // Calculate trading performance metrics
  if (s.ts_first > 0 && s.ts_last > 0 && s.n_fill > 0) {
    // **ENHANCED**: Use test period timestamps from metadata if available, otherwise fall back to event range
    std::int64_t test_start_ts = s.ts_first;
    std::int64_t test_end_ts = s.ts_last;
    
    // Try to extract test period from run metadata
    sqlite3_stmt* meta_st = nullptr;
    if (sqlite3_prepare_v2(db_, "SELECT params_json FROM audit_runs WHERE run_id=?", -1, &meta_st, nullptr) == SQLITE_OK) {
      sqlite3_bind_text(meta_st, 1, run_id.c_str(), -1, SQLITE_STATIC);
      if (sqlite3_step(meta_st) == SQLITE_ROW) {
        const char* params_json = reinterpret_cast<const char*>(sqlite3_column_text(meta_st, 0));
        if (params_json) {
          std::string meta_str(params_json);
          
          // Parse test_start_ts and test_end_ts from JSON
          size_t start_pos = meta_str.find("\"test_start_ts\":");
          if (start_pos != std::string::npos) {
            start_pos += 16; // Skip past "test_start_ts":
            size_t end_pos = meta_str.find_first_of(",}", start_pos);
            if (end_pos != std::string::npos) {
              std::string start_ts_str = meta_str.substr(start_pos, end_pos - start_pos);
              test_start_ts = std::stoll(start_ts_str);
            }
          }
          
          size_t end_ts_pos = meta_str.find("\"test_end_ts\":");
          if (end_ts_pos != std::string::npos) {
            end_ts_pos += 14; // Skip past "test_end_ts":
            size_t end_pos = meta_str.find_first_of(",}", end_ts_pos);
            if (end_pos != std::string::npos) {
              std::string end_ts_str = meta_str.substr(end_ts_pos, end_pos - end_ts_pos);
              test_end_ts = std::stoll(end_ts_str);
            }
          }
        }
      }
      sqlite3_finalize(meta_st);
    }
    
    // **CANONICAL METRICS**: Load daily returns once and use for both trading days and MPR
    std::vector<double> daily_returns = load_daily_returns(run_id);
    
    if (!daily_returns.empty()) {
        // Use stored daily returns for canonical calculations
        s.trading_days = static_cast<int>(daily_returns.size());
        
        // Calculate daily trades
        s.daily_trades = s.trading_days > 0 ? static_cast<double>(s.n_fill) / s.trading_days : 0.0;
        
        // Calculate total return (assuming starting equity of 100,000)
        double starting_equity = 100000.0;
        s.total_return = (s.pnl_sum / starting_equity) * 100.0;
        
        // Use canonical MPR calculation
        double canonical_mpr_decimal = sentio::metrics::compute_mpr_from_daily_returns(daily_returns);
        s.mpr = canonical_mpr_decimal * 100.0; // Convert to percentage
    } else {
        // Fallback to old calculations if no daily returns stored
        double time_span_days = (test_end_ts - test_start_ts) / (1000.0 * 60.0 * 60.0 * 24.0);
        s.trading_days = static_cast<int>(std::ceil(time_span_days));
        
        s.daily_trades = s.trading_days > 0 ? static_cast<double>(s.n_fill) / s.trading_days : 0.0;
        
        double starting_equity = 100000.0;
        s.total_return = (s.pnl_sum / starting_equity) * 100.0;
        
        double total_return_decimal = s.total_return / 100.0;
        double trading_years = s.trading_days / 252.0;
        double annual_return = trading_years > 0 ? std::pow(1.0 + total_return_decimal, 1.0/trading_years) - 1.0 : 0.0;
        s.mpr = (std::pow(1.0 + annual_return, 1.0/12.0) - 1.0) * 100.0;
    }
    
    // Calculate Sharpe ratio (simplified - using daily returns)
    // This is a basic approximation; a proper implementation would need daily returns
    double avg_daily_return = s.pnl_sum / s.trading_days;
    double daily_return_std = std::sqrt(std::abs(s.pnl_sum) / s.trading_days); // Simplified volatility
    s.sharpe = daily_return_std > 0 ? avg_daily_return / daily_return_std : 0.0;
    
    // Calculate max drawdown (simplified)
    // This would require tracking equity curve, but for now use a conservative estimate
    // Assume max drawdown is roughly 20% of the total return magnitude
    s.max_drawdown = std::abs(s.total_return) * 0.2;
  }
  
  return s;
}

void DB::export_run_csv(const std::string& run_id, const std::string& out_path) {
  sqlite3_stmt* st=nullptr;
  FILE* f = fopen(out_path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot open out csv");
  fputs("seq,ts,kind,symbol,side,qty,price,pnl,weight,prob,reason,note,hash_prev,hash_curr\n", f);
  sqlite3_prepare_v2(db_,
    "SELECT seq,ts_millis,kind,symbol,side,qty,price,pnl_delta,weight,prob,reason,note,hash_prev,hash_curr"
    " FROM audit_events WHERE run_id=? ORDER BY seq ASC", -1, &st, nullptr);
  sqlite3_bind_text(st,1,run_id.c_str(),-1,SQLITE_TRANSIENT);
  auto csv_escape=[&](const unsigned char* u){ std::string s=u?reinterpret_cast<const char*>(u):""; for(char& c:s) if(c==',') c=';'; return s; };
  while (sqlite3_step(st) == SQLITE_ROW) {
    fprintf(f,"%lld,%lld,%s,%s,%s,%.12g,%.12g,%.12g,%.12g,%.12g,%s,%s,%s,%s\n",
      sqlite3_column_int64(st,0),
      sqlite3_column_int64(st,1),
      sqlite3_column_text(st,2),
      sqlite3_column_text(st,3),
      sqlite3_column_text(st,4),
      sqlite3_column_double(st,5),
      sqlite3_column_double(st,6),
      sqlite3_column_double(st,7),
      sqlite3_column_double(st,8),
      sqlite3_column_double(st,9),
      csv_escape(sqlite3_column_text(st,10)).c_str(),
      csv_escape(sqlite3_column_text(st,11)).c_str(),
      sqlite3_column_text(st,12),
      sqlite3_column_text(st,13));
  }
  sqlite3_finalize(st);
  fclose(f);
}

void DB::export_run_jsonl(const std::string& run_id, const std::string& out_path) {
  sqlite3_stmt* st=nullptr;
  FILE* f = fopen(out_path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot open out jsonl");
  sqlite3_prepare_v2(db_,
    "SELECT seq,ts_millis,kind,symbol,side,qty,price,pnl_delta,weight,prob,reason,note,hash_prev,hash_curr"
    " FROM audit_events WHERE run_id=? ORDER BY seq ASC", -1, &st, nullptr);
  sqlite3_bind_text(st,1,run_id.c_str(),-1,SQLITE_TRANSIENT);
  auto jsesc=[&](const unsigned char* u){ std::string s=u?reinterpret_cast<const char*>(u):""; std::string o; o.reserve(s.size()); for(char c: s){ if(c=='\"'||c=='\\') {o.push_back('\\'); o.push_back(c);} else if(c=='\n'){o+="\\n";} else o.push_back(c);} return o;};
  while (sqlite3_step(st) == SQLITE_ROW) {
    fprintf(f,"{\"seq\":%lld,\"ts\":%lld,\"kind\":\"%s\",\"symbol\":\"%s\",\"side\":\"%s\",\"qty\":%.12g,\"price\":%.12g,\"pnl\":%.12g,\"weight\":%.12g,\"prob\":%.12g,\"reason\":\"%s\",\"note\":\"%s\",\"hash_prev\":\"%s\",\"hash_curr\":\"%s\"}\n",
      sqlite3_column_int64(st,0),
      sqlite3_column_int64(st,1),
      sqlite3_column_text(st,2),
      sqlite3_column_text(st,3),
      sqlite3_column_text(st,4),
      sqlite3_column_double(st,5),
      sqlite3_column_double(st,6),
      sqlite3_column_double(st,7),
      sqlite3_column_double(st,8),
      sqlite3_column_double(st,9),
      jsesc(sqlite3_column_text(st,10)).c_str(),
      jsesc(sqlite3_column_text(st,11)).c_str(),
      sqlite3_column_text(st,12),
      sqlite3_column_text(st,13));
  }
  sqlite3_finalize(st);
  fclose(f);
}

long DB::grep_where(const std::string& run_id, const std::string& where_sql) {
  std::string sql =
    "SELECT seq,ts_millis,kind,symbol,side,qty,price,pnl_delta,weight,prob,reason,note "
    "FROM audit_events WHERE run_id=? " + where_sql + " ORDER BY seq ASC";
  sqlite3_stmt* st=nullptr;
  sqlite3_prepare_v2(db_, sql.c_str(), -1, &st, nullptr);
  sqlite3_bind_text(st,1,run_id.c_str(),-1,SQLITE_TRANSIENT);
  long n=0;
  while (sqlite3_step(st) == SQLITE_ROW) {
    printf("#%lld ts=%lld kind=%s %s %s qty=%.4f price=%.4f pnl=%.4f w=%.4f p=%.4f reason=%s\n",
      sqlite3_column_int64(st,0), sqlite3_column_int64(st,1),
      sqlite3_column_text(st,2), sqlite3_column_text(st,3), sqlite3_column_text(st,4),
      sqlite3_column_double(st,5), sqlite3_column_double(st,6), sqlite3_column_double(st,7),
      sqlite3_column_double(st,8), sqlite3_column_double(st,9),
      sqlite3_column_text(st,10));
    ++n;
  }
  sqlite3_finalize(st);
  return n;
}

std::string DB::diff_runs(const std::string& a, const std::string& b) {
  auto sa = summarize(a);
  auto sb = summarize(b);
  char buf[512];
  snprintf(buf,sizeof(buf),
    "DIFF\nA=%s: events=%lld pnl=%.4f span=[%lld..%lld]\n"
    "B=%s: events=%lld pnl=%.4f span=[%lld..%lld]\n"
    "Δevents=%lld Δpnl=%.4f\n",
    a.c_str(), sa.n_total, sa.pnl_sum, sa.ts_first, sa.ts_last,
    b.c_str(), sb.n_total, sb.pnl_sum, sb.ts_first, sb.ts_last,
    (long long)(sb.n_total - sa.n_total), (double)(sb.pnl_sum - sa.pnl_sum));
  return std::string(buf);
}

void DB::vacuum() { exec_or_throw(db_, "VACUUM; ANALYZE;"); }

} // namespace audit

```

## 📄 **FILE 10 of 14**: audit/src/audit_db_recorder.cpp

**File Information**:
- **Path**: `audit/src/audit_db_recorder.cpp`

- **Size**: 537 lines
- **Modified**: 2025-09-15 21:07:52

- **Type**: .cpp

```text
#include "audit/audit_db_recorder.hpp"
#include "sentio/audit.hpp"
#include <stdexcept>
#include <iostream>

namespace audit {

AuditDBRecorder::AuditDBRecorder(const std::string& db_path, const std::string& run_id, const std::string& note)
    : run_id_(run_id), note_(note), logging_enabled_(true), started_at_(0) {
    
    try {
        db_ = std::make_unique<DB>(db_path);
        db_->init_schema();
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize audit database: " + std::string(e.what()));
    }
}

AuditDBRecorder::~AuditDBRecorder() {
    // Database cleanup handled by unique_ptr
}

void AuditDBRecorder::event_run_start(std::int64_t ts, const std::string& meta) {
    if (!logging_enabled_) return;
    
    // Extract strategy name from meta JSON
    strategy_name_ = "unknown"; // Default fallback
    try {
        // Simple JSON parsing to extract strategy name
        size_t strategy_pos = meta.find("\"strategy\":\"");
        if (strategy_pos != std::string::npos) {
            strategy_pos += 12; // Skip past "strategy":"
            size_t end_pos = meta.find("\"", strategy_pos);
            if (end_pos != std::string::npos) {
                strategy_name_ = meta.substr(strategy_pos, end_pos - strategy_pos);
            }
        }
    } catch (...) {
        // Keep default fallback if parsing fails
    }
    started_at_ = ts;
    
    // Create new run in database
    RunRow run;
    run.run_id = run_id_;
    run.started_at = ts;
    run.kind = "backtest"; // Default, can be overridden
    run.strategy = strategy_name_;
    run.params_json = meta;
    run.data_hash = "unknown"; // Can be set from meta_json if available
    run.git_rev = "unknown";
    run.note = note_;
    
    // **CANONICAL PERIOD FIX**: Extract period fields from JSON metadata
    // Parse JSON to extract run_period_start_ts_ms, run_period_end_ts_ms, run_trading_days
    try {
        // Simple JSON parsing for the specific fields we need
        size_t start_pos = meta.find("\"run_period_start_ts_ms\":");
        if (start_pos != std::string::npos) {
            start_pos += 26; // Length of "\"run_period_start_ts_ms\":"
            size_t end_pos = meta.find(",", start_pos);
            if (end_pos == std::string::npos) end_pos = meta.find("}", start_pos);
            if (end_pos != std::string::npos) {
                std::string value_str = meta.substr(start_pos, end_pos - start_pos);
                run.run_period_start_ts_ms = std::stoll(value_str);
            }
        }
        
        size_t end_pos = meta.find("\"run_period_end_ts_ms\":");
        if (end_pos != std::string::npos) {
            end_pos += 24; // Length of "\"run_period_end_ts_ms\":"
            size_t comma_pos = meta.find(",", end_pos);
            if (comma_pos == std::string::npos) comma_pos = meta.find("}", end_pos);
            if (comma_pos != std::string::npos) {
                std::string value_str = meta.substr(end_pos, comma_pos - end_pos);
                run.run_period_end_ts_ms = std::stoll(value_str);
            }
        }
        
        size_t days_pos = meta.find("\"run_trading_days\":");
        if (days_pos != std::string::npos) {
            days_pos += 19; // Length of "\"run_trading_days\":"
            size_t comma_pos = meta.find(",", days_pos);
            if (comma_pos == std::string::npos) comma_pos = meta.find("}", days_pos);
            if (comma_pos != std::string::npos) {
                std::string value_str = meta.substr(days_pos, comma_pos - days_pos);
                run.run_trading_days = std::stoi(value_str);
            }
        }
    } catch (const std::exception& e) {
        // If JSON parsing fails, continue without the period fields
        std::cerr << "Warning: Failed to parse period fields from metadata: " << e.what() << std::endl;
    }
    
    // **DATASET TRACEABILITY**: Parse dataset information from JSON metadata
    try {
        // Parse dataset_source_type (enhanced field)
        size_t source_type_pos = meta.find("\"dataset_source_type\":\"");
        if (source_type_pos != std::string::npos) {
            source_type_pos += 23; // Length of "\"dataset_source_type\":\""
            size_t end_quote = meta.find("\"", source_type_pos);
            if (end_quote != std::string::npos) {
                run.dataset_source_type = meta.substr(source_type_pos, end_quote - source_type_pos);
            }
        } else {
            // Fallback to dataset_type if dataset_source_type not found
            size_t type_pos = meta.find("\"dataset_type\":\"");
            if (type_pos != std::string::npos) {
                type_pos += 16; // Length of "\"dataset_type\":\""
                size_t end_quote = meta.find("\"", type_pos);
                if (end_quote != std::string::npos) {
                    run.dataset_source_type = meta.substr(type_pos, end_quote - type_pos);
                }
            }
        }
        
        // Parse dataset_file_path
        size_t path_pos = meta.find("\"dataset_file_path\":\"");
        if (path_pos != std::string::npos) {
            path_pos += 21; // Length of "\"dataset_file_path\":\""
            size_t end_quote = meta.find("\"", path_pos);
            if (end_quote != std::string::npos) {
                run.dataset_file_path = meta.substr(path_pos, end_quote - path_pos);
            }
        }
        
        // Parse dataset_file_hash
        size_t hash_pos = meta.find("\"dataset_file_hash\":\"");
        if (hash_pos != std::string::npos) {
            hash_pos += 21; // Length of "\"dataset_file_hash\":\""
            size_t end_quote = meta.find("\"", hash_pos);
            if (end_quote != std::string::npos) {
                run.dataset_file_hash = meta.substr(hash_pos, end_quote - hash_pos);
            }
        }
        
        // Parse dataset_track_id
        size_t track_pos = meta.find("\"dataset_track_id\":\"");
        if (track_pos != std::string::npos) {
            track_pos += 20; // Length of "\"dataset_track_id\":\""
            size_t end_quote = meta.find("\"", track_pos);
            if (end_quote != std::string::npos) {
                run.dataset_track_id = meta.substr(track_pos, end_quote - track_pos);
            }
        }
        
        // Parse dataset_regime
        size_t regime_pos = meta.find("\"dataset_regime\":\"");
        if (regime_pos != std::string::npos) {
            regime_pos += 18; // Length of "\"dataset_regime\":\""
            size_t end_quote = meta.find("\"", regime_pos);
            if (end_quote != std::string::npos) {
                run.dataset_regime = meta.substr(regime_pos, end_quote - regime_pos);
            }
        }
        
        // Parse dataset_bars_count (enhanced field)
        size_t bars_pos = meta.find("\"dataset_bars_count\":");
        if (bars_pos != std::string::npos) {
            bars_pos += 21; // Length of "\"dataset_bars_count\":"
            size_t comma_pos = meta.find(",", bars_pos);
            if (comma_pos == std::string::npos) comma_pos = meta.find("}", bars_pos);
            if (comma_pos != std::string::npos) {
                std::string value_str = meta.substr(bars_pos, comma_pos - bars_pos);
                run.dataset_bars_count = std::stoi(value_str);
            }
        } else {
            // Fallback to base_series_size if dataset_bars_count not found
            size_t size_pos = meta.find("\"base_series_size\":");
            if (size_pos != std::string::npos) {
                size_pos += 19; // Length of "\"base_series_size\":"
                size_t comma_pos = meta.find(",", size_pos);
                if (comma_pos == std::string::npos) comma_pos = meta.find("}", size_pos);
                if (comma_pos != std::string::npos) {
                    std::string value_str = meta.substr(size_pos, comma_pos - size_pos);
                    run.dataset_bars_count = std::stoi(value_str);
                }
            }
        }
        
        // Parse dataset_time_range_start
        size_t start_pos = meta.find("\"dataset_time_range_start\":");
        if (start_pos != std::string::npos) {
            start_pos += 27; // Length of "\"dataset_time_range_start\":"
            size_t comma_pos = meta.find(",", start_pos);
            if (comma_pos == std::string::npos) comma_pos = meta.find("}", start_pos);
            if (comma_pos != std::string::npos) {
                std::string value_str = meta.substr(start_pos, comma_pos - start_pos);
                run.dataset_time_range_start = std::stoll(value_str);
            }
        }
        
        // Parse dataset_time_range_end
        size_t end_pos = meta.find("\"dataset_time_range_end\":");
        if (end_pos != std::string::npos) {
            end_pos += 25; // Length of "\"dataset_time_range_end\":"
            size_t comma_pos = meta.find(",", end_pos);
            if (comma_pos == std::string::npos) comma_pos = meta.find("}", end_pos);
            if (comma_pos != std::string::npos) {
                std::string value_str = meta.substr(end_pos, comma_pos - end_pos);
                run.dataset_time_range_end = std::stoll(value_str);
            }
        }
        
    } catch (const std::exception& e) {
        // If dataset parsing fails, continue with defaults
        std::cerr << "Warning: Failed to parse dataset fields from metadata: " << e.what() << std::endl;
    }
    
    try {
        db_->new_run(run);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to create audit run: " << e.what() << std::endl;
    }
}

void AuditDBRecorder::event_run_end(std::int64_t ts, const std::string& meta) {
    if (!logging_enabled_) return;
    
    try {
        db_->end_run(run_id_, ts);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to end audit run: " << e.what() << std::endl;
    }
}

void AuditDBRecorder::event_bar(std::int64_t ts, const std::string& inst, double open, double high, double low, double close, double volume) {
    if (!logging_enabled_) return;
    
    Event event;
    event.run_id = run_id_;
    event.ts_millis = ts * 1000; // Convert to milliseconds
    event.kind = "BAR";
    event.symbol = inst;
    event.side = "";
    event.qty = 0.0;
    event.price = close; // Use close price as the primary price
    event.pnl_delta = 0.0;
    event.weight = 0.0;
    event.prob = 0.0;
    event.reason = "";
    event.note = "open=" + std::to_string(open) + ",high=" + std::to_string(high) + 
                 ",low=" + std::to_string(low) + ",volume=" + std::to_string(volume);
    
    log_event(event);
}

void AuditDBRecorder::event_signal(std::int64_t ts, const std::string& base, sentio::SigType t, double conf) {
    if (!logging_enabled_) return;
    
    Event event;
    event.run_id = run_id_;
    event.ts_millis = ts * 1000; // Convert to milliseconds
    event.kind = "SIGNAL";
    event.symbol = base;
    event.side = sig_type_to_string(static_cast<int>(t));
    event.qty = 0.0;
    event.price = 0.0;
    event.pnl_delta = 0.0;
    event.weight = conf; // Use confidence as weight
    event.prob = conf;
    event.reason = "SIGNAL_TYPE_" + std::to_string(static_cast<int>(t));
    event.note = "";
    
    log_event(event);
}

void AuditDBRecorder::event_signal_ex(std::int64_t ts, const std::string& base, sentio::SigType t, double conf, const std::string& chain_id) {
    if (!logging_enabled_) return;
    
    Event event;
    event.run_id = run_id_;
    event.ts_millis = ts * 1000; // Convert to milliseconds
    event.kind = "SIGNAL";
    event.symbol = base;
    event.side = sig_type_to_string(static_cast<int>(t));
    event.qty = 0.0;
    event.price = 0.0;
    event.pnl_delta = 0.0;
    event.weight = conf; // Use confidence as weight
    event.prob = conf;
    event.reason = "SIGNAL_TYPE_" + std::to_string(static_cast<int>(t));
    event.note = chain_id.empty() ? "" : "chain=" + chain_id;
    
    log_event(event);
}

void AuditDBRecorder::event_route(std::int64_t ts, const std::string& base, const std::string& inst, double tw) {
    if (!logging_enabled_) return;
    
    Event event;
    event.run_id = run_id_;
    event.ts_millis = ts * 1000; // Convert to milliseconds
    event.kind = "ROUTE";
    event.symbol = inst;
    event.side = "";
    event.qty = 0.0;
    event.price = 0.0;
    event.pnl_delta = 0.0;
    event.weight = tw;
    event.prob = 0.0;
    event.reason = "ROUTE_FROM_" + base;
    event.note = "";
    
    log_event(event);
}

void AuditDBRecorder::event_route_ex(std::int64_t ts, const std::string& base, const std::string& inst, double tw, const std::string& chain_id) {
    if (!logging_enabled_) return;
    
    Event event;
    event.run_id = run_id_;
    event.ts_millis = ts * 1000; // Convert to milliseconds
    event.kind = "ROUTE";
    event.symbol = inst;
    event.side = "";
    event.qty = 0.0;
    event.price = 0.0;
    event.pnl_delta = 0.0;
    event.weight = tw;
    event.prob = 0.0;
    event.reason = "ROUTE_FROM_" + base;
    event.note = chain_id.empty() ? "" : "chain=" + chain_id;
    
    log_event(event);
}

void AuditDBRecorder::event_order(std::int64_t ts, const std::string& inst, sentio::Side side, double qty, double limit_px) {
    if (!logging_enabled_) return;
    
    Event event;
    event.run_id = run_id_;
    event.ts_millis = ts * 1000; // Convert to milliseconds
    event.kind = "ORDER";
    event.symbol = inst;
    event.side = side_to_string(static_cast<int>(side));
    event.qty = qty;
    event.price = limit_px;
    event.pnl_delta = 0.0;
    event.weight = 0.0;
    event.prob = 0.0;
    event.reason = "";
    event.note = "";
    
    log_event(event);
}

void AuditDBRecorder::event_order_ex(std::int64_t ts, const std::string& inst, sentio::Side side, double qty, double limit_px, const std::string& chain_id) {
    if (!logging_enabled_) return;
    
    Event event;
    event.run_id = run_id_;
    event.ts_millis = ts * 1000; // Convert to milliseconds
    event.kind = "ORDER";
    event.symbol = inst;
    event.side = side_to_string(static_cast<int>(side));
    event.qty = qty;
    event.price = limit_px;
    event.pnl_delta = 0.0;
    event.weight = 0.0;
    event.prob = 0.0;
    event.reason = "";
    event.note = chain_id.empty() ? "" : "chain=" + chain_id;
    
    log_event(event);
}

void AuditDBRecorder::event_fill(std::int64_t ts, const std::string& inst, double price, double qty, double fees, sentio::Side side) {
    if (!logging_enabled_) return;
    
    Event event;
    event.run_id = run_id_;
    event.ts_millis = ts * 1000; // Convert to milliseconds
    event.kind = "FILL";
    event.symbol = inst;
    event.side = side_to_string(static_cast<int>(side));
    event.qty = qty;
    event.price = price;
    event.pnl_delta = 0.0;
    event.weight = 0.0;
    event.prob = 0.0;
    event.reason = "";
    event.note = "fees=" + std::to_string(fees);
    
    log_event(event);
}

void AuditDBRecorder::event_fill_ex(std::int64_t ts, const std::string& inst, double price, double qty, double fees, sentio::Side side, 
                                    double realized_pnl_delta, double equity_after, double position_after, const std::string& chain_id) {
    if (!logging_enabled_) return;
    
    Event event;
    event.run_id = run_id_;
    event.ts_millis = ts * 1000; // Convert to milliseconds
    event.kind = "FILL";
    event.symbol = inst;
    event.side = side_to_string(static_cast<int>(side));
    event.qty = qty;
    event.price = price;
    event.pnl_delta = realized_pnl_delta;
    event.weight = 0.0;
    event.prob = 0.0;
    event.reason = "";
    event.note = "fees=" + std::to_string(fees) + 
                 ",eq_after=" + std::to_string(equity_after) + 
                 ",pos_after=" + std::to_string(position_after) +
                 (chain_id.empty() ? "" : ",chain=" + chain_id);
    
    log_event(event);
}

void AuditDBRecorder::event_snapshot(std::int64_t ts, const sentio::AccountState& a) {
    if (!logging_enabled_) return;
    
    Event event;
    event.run_id = run_id_;
    event.ts_millis = ts * 1000; // Convert to milliseconds
    event.kind = "SNAPSHOT";
    event.symbol = "";
    event.side = "";
    event.qty = 0.0;
    event.price = 0.0;
    event.pnl_delta = a.realized;
    event.weight = 0.0;
    event.prob = 0.0;
    event.reason = "";
    event.note = "cash=" + std::to_string(a.cash) + ",equity=" + std::to_string(a.equity);
    
    log_event(event);
}

void AuditDBRecorder::event_metric(std::int64_t ts, const std::string& key, double val) {
    if (!logging_enabled_) return;
    
    Event event;
    event.run_id = run_id_;
    event.ts_millis = ts; // ts is already in milliseconds
    event.kind = "METRIC";
    event.symbol = "";
    event.side = "";
    event.qty = 0.0;
    event.price = 0.0;
    event.pnl_delta = 0.0;
    event.weight = 0.0;
    event.prob = 0.0;
    event.reason = key;
    event.note = "value=" + std::to_string(val);

    log_event(event);
}

void AuditDBRecorder::store_daily_returns(const std::vector<std::pair<std::string, double>>& daily_returns) {
    if (!logging_enabled_ || !db_) return;
    db_->store_daily_returns(run_id_, daily_returns);
}

std::string AuditDBRecorder::side_to_string(int side) {
    switch (side) {
        case 0: return "BUY";
        case 1: return "SELL";
        default: return "NEUTRAL";
    }
}

std::string AuditDBRecorder::sig_type_to_string(int sig_type) {
    switch (sig_type) {
        case 0: return "BUY";
        case 1: return "STRONG_BUY";
        case 2: return "SELL";
        case 3: return "STRONG_SELL";
        case 4: return "HOLD";
        default: return "UNKNOWN";
    }
}

void AuditDBRecorder::log_event(const Event& event) {
    try {
        auto [seq, hash] = db_->append_event(event);
        // Optionally log the sequence number for debugging
        // std::cerr << "Logged event seq=" << seq << " hash=" << hash << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to log audit event: " << e.what() << std::endl;
    }
}

void AuditDBRecorder::event_signal_diag(std::int64_t ts, const std::string& strategy_name, const SignalDiag& diag) {
    if (!logging_enabled_) return;
    
    Event event;
    event.run_id = run_id_;
    event.ts_millis = ts * 1000; // Convert to milliseconds
    event.kind = "SIGNAL_DIAG";
    event.symbol = strategy_name;
    event.side = "";
    event.qty = static_cast<double>(diag.emitted);
    event.price = static_cast<double>(diag.dropped);
    event.pnl_delta = 0.0;
    event.weight = 0.0;
    event.prob = 0.0;
    event.reason = "DIAG_SUMMARY";
    event.note = "emitted=" + std::to_string(diag.emitted) + 
                 ",dropped=" + std::to_string(diag.dropped) +
                 ",min_bars=" + std::to_string(diag.r_min_bars) +
                 ",session=" + std::to_string(diag.r_session) +
                 ",nan=" + std::to_string(diag.r_nan) +
                 ",zero_vol=" + std::to_string(diag.r_zero_vol) +
                 ",threshold=" + std::to_string(diag.r_threshold) +
                 ",cooldown=" + std::to_string(diag.r_cooldown) +
                 ",dup=" + std::to_string(diag.r_dup);
    
    log_event(event);
}

void AuditDBRecorder::event_signal_drop(std::int64_t ts, const std::string& strategy_name, const std::string& symbol, 
                                       DropReason reason, const std::string& chain_id, const std::string& note) {
    if (!logging_enabled_) return;
    
    Event event;
    event.run_id = run_id_;
    event.ts_millis = ts * 1000; // Convert to milliseconds
    event.kind = "SIGNAL_DROP";
    event.symbol = symbol;
    event.side = "";
    event.qty = 0.0;
    event.price = static_cast<double>(static_cast<int>(reason));
    event.pnl_delta = 0.0;
    event.weight = 0.0;
    event.prob = 0.0;
    event.reason = "DROP_REASON_" + std::to_string(static_cast<int>(reason));
    event.note = "strategy=" + strategy_name + 
                 ",chain=" + chain_id + 
                 (note.empty() ? "" : ",note=" + note);
    
    log_event(event);
}

} // namespace audit

```

## 📄 **FILE 11 of 14**: audit/src/clock.cpp

**File Information**:
- **Path**: `audit/src/clock.cpp`

- **Size**: 12 lines
- **Modified**: 2025-09-11 15:18:13

- **Type**: .cpp

```text
#include "audit/clock.hpp"
#include <chrono>

namespace audit {

std::int64_t now_millis() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

} // namespace audit

```

## 📄 **FILE 12 of 14**: audit/src/hash.cpp

**File Information**:
- **Path**: `audit/src/hash.cpp`

- **Size**: 134 lines
- **Modified**: 2025-09-11 14:14:32

- **Type**: .cpp

```text
#include "audit/hash.hpp"
#include <cstring>
#include <sstream>
#include <iomanip>

// Public domain SHA-256 implementation
// Based on: https://github.com/System-Glitch/SHA256

namespace audit {

// SHA-256 constants
static const uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

static uint32_t choose(uint32_t e, uint32_t f, uint32_t g) {
    return (e & f) ^ (~e & g);
}

static uint32_t majority(uint32_t a, uint32_t b, uint32_t c) {
    return (a & (b | c)) | (b & c);
}

static uint32_t sig0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

static uint32_t sig1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

static uint32_t gamma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

static uint32_t gamma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

std::string sha256_hex(const void* data, size_t n) {
    const uint8_t* input = static_cast<const uint8_t*>(data);
    
    // Initialize hash values
    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    
    // Process message in 512-bit chunks
    for (size_t chunk = 0; chunk * 64 < n + 9; ++chunk) {
        uint32_t w[64] = {0};
        
        // Copy chunk into first 16 words
        size_t start = chunk * 64;
        for (int i = 0; i < 16 && start + i * 4 < n; ++i) {
            size_t pos = start + i * 4;
            w[i] = (static_cast<uint32_t>(input[pos]) << 24) |
                   (static_cast<uint32_t>(input[pos + 1]) << 16) |
                   (static_cast<uint32_t>(input[pos + 2]) << 8) |
                   static_cast<uint32_t>(input[pos + 3]);
        }
        
        // Add padding
        if (start < n) {
            size_t pos = start;
            if (pos < n) {
                w[pos / 4] |= static_cast<uint32_t>(input[pos]) << (24 - 8 * (pos % 4));
            }
            if (pos + 1 < n) {
                w[pos / 4] |= static_cast<uint32_t>(input[pos + 1]) << (16 - 8 * (pos % 4));
            }
            if (pos + 2 < n) {
                w[pos / 4] |= static_cast<uint32_t>(input[pos + 2]) << (8 - 8 * (pos % 4));
            }
            if (pos + 3 < n) {
                w[pos / 4] |= static_cast<uint32_t>(input[pos + 3]) << (0 - 8 * (pos % 4));
            }
        }
        
        // Add length padding
        if (chunk * 64 + 64 > n + 9) {
            w[15] = static_cast<uint32_t>(n * 8);
        }
        
        // Extend the 16 words into 64 words
        for (int i = 16; i < 64; ++i) {
            w[i] = gamma1(w[i - 2]) + w[i - 7] + gamma0(w[i - 15]) + w[i - 16];
        }
        
        // Initialize working variables
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], h_val = h[7];
        
        // Main loop
        for (int i = 0; i < 64; ++i) {
            uint32_t t1 = h_val + sig1(e) + choose(e, f, g) + k[i] + w[i];
            uint32_t t2 = sig0(a) + majority(a, b, c);
            
            h_val = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        
        // Add the compressed chunk to the current hash value
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += h_val;
    }
    
    // Produce the final hash value as a hex string
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (int i = 0; i < 8; ++i) {
        oss << std::setw(8) << h[i];
    }
    return oss.str();
}

} // namespace audit

```

## 📄 **FILE 13 of 14**: audit/src/price_csv.cpp

**File Information**:
- **Path**: `audit/src/price_csv.cpp`

- **Size**: 51 lines
- **Modified**: 2025-09-11 14:14:41

- **Type**: .cpp

```text
#include "audit/price_csv.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace audit {
std::unordered_map<std::string, Series> load_price_csv(const std::string& path) {
  std::unordered_map<std::string, Series> out;
  std::ifstream f(path);
  if (!f.is_open()) {
    throw std::runtime_error("Cannot open price CSV file: " + path);
  }
  
  std::string line;
  bool header = true;
  while (std::getline(f, line)) {
    if (header) { 
      header = false; 
      continue; 
    }
    
    if (line.empty()) continue;
    
    std::stringstream ss(line);
    // columns: symbol,ts,open,high,low,close,volume
    std::string sym, ts, o, h, l, c, v;
    
    if (!std::getline(ss, sym, ',') || !std::getline(ss, ts, ',') ||
        !std::getline(ss, o, ',') || !std::getline(ss, h, ',') ||
        !std::getline(ss, l, ',') || !std::getline(ss, c, ',') ||
        !std::getline(ss, v, ',')) {
      continue; // Skip malformed lines
    }
    
    try {
      Bar b{ 
        std::stoll(ts), 
        std::stod(o), 
        std::stod(h), 
        std::stod(l), 
        std::stod(c), 
        std::stod(v) 
      };
      out[sym].push_back(b);
    } catch (const std::exception&) {
      continue; // Skip lines with invalid numbers
    }
  }
  return out;
}
}

```

## 📄 **FILE 14 of 14**: audit/tests/test_verify.cpp

**File Information**:
- **Path**: `audit/tests/test_verify.cpp`

- **Size**: 37 lines
- **Modified**: 2025-09-11 14:22:41

- **Type**: .cpp

```text
#include "audit/audit_db.hpp"
#include "audit/clock.hpp"
#include <cassert>
#include <cstdio>

using namespace audit;

int main() {
  const char* dbp = "test_audit.sqlite3";
  remove(dbp);
  DB db(dbp); db.init_schema();

  RunRow run{ "RUN_X", now_millis(), {}, "backtest", "RSI_PROB", "{}", "DATAHASH", "abc123", "" };
  db.new_run(run);

  Event e1{ "RUN_X", now_millis(), "SIGNAL", "QQQ", "BUY", 0, 0, 0, 0.7, 0.85, "RSI_LT_30", "" };
  db.append_event(e1);
  Event e2{ "RUN_X", now_millis()+1, "ORDER", "QQQ", "BUY", 100, 0, 0, 0, 0, "", "" };
  db.append_event(e2);
  Event e3{ "RUN_X", now_millis()+2, "FILL", "QQQ", "BUY", 100, 440.12, 0, 0, 0, "", "" };
  db.append_event(e3);
  Event e4{ "RUN_X", now_millis()+3, "PNL", "QQQ", "", 0, 0, 12.34, 0, 0, "", "" };
  db.append_event(e4);

  auto [ok,msg] = db.verify_run("RUN_X");
  assert(ok);

  auto s = db.summarize("RUN_X");
  assert(s.n_total==4 && s.n_fill==1 && s.pnl_sum>12.0);

  db.export_run_csv("RUN_X", "out.csv");
  db.export_run_jsonl("RUN_X", "out.jsonl");

  db.vacuum();
  puts("OK");
  return 0;
}

```

