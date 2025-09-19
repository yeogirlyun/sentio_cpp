# Position Accumulation Bug Fix

**Generated**: 2025-09-19 13:06:02
**Source Directory**: /Users/yeogirlyun/C++/sentio_cpp
**Description**: Critical bug fix for audit system recording wrong trade quantities

**Total Files**: 4

---

## 🐛 **BUG REPORT**

# CRITICAL BUG REPORT: Position Accumulation - Audit vs StratTest Discrepancy

## 🚨 EXECUTIVE SUMMARY

**CRITICAL FINDING**: After database reset, the audit system STILL shows massive position accumulation and conflicting positions. This indicates a real, current bug in either:
1. The audit recording system (recording wrong data)
2. The strattest execution system (executing wrong trades)
3. A disconnect between what strattest reports vs what actually executes

## 📊 EVIDENCE FROM FRESH DATABASE

### **Audit System Output (Run ID: 225376)**
```
│ Cash Balance        │ $-740108.71 │ Position Value      │ $ 840691.85 │
│ PSQ        │      599 │ $     80239.05 │
│ QQQ        │      432 │ $    253686.36 │
│ SQQQ       │     9338 │ $    169410.91 │
│ TQQQ       │      215 │ $    337355.52 │
│ Total Trades        │        3216 │
```

### **Trade Pattern Analysis**
```
│ TQQQ   │ 🟢BUY  │    0.327 │ → TQQQ:236 shares
│ TQQQ   │ 🔴SELL │    0.307 │ → TQQQ:235 shares  
│ TQQQ   │ 🟢BUY  │    0.294 │ → TQQQ:236 shares
```

**CRITICAL ISSUE**: Tiny trades (0.3 shares) somehow result in massive positions (236 shares)!

## 🔍 ROOT CAUSE IDENTIFIED!

### **✅ STRATTEST EXECUTION IS CORRECT**
**Debug Output Confirms:**
- Position updates work perfectly: `[APPLY_FILL] SID:0 | Final: 0`
- Final positions: QQQ: 0 shares, TQQQ: 0 shares (properly closed at EOD)
- Reasonable trade sizes: 55.0563 shares, 20.6575 shares
- Total fills: 304 (matches strattest report)

### **🚨 AUDIT RECORDING BUG FOUND**
**Line 195 in `src/runner.cpp`:**
```cpp
audit.event_fill_ex(bar.ts_utc_epoch, decision.instrument, 
                  instrument_price, std::abs(trade_qty), fees, side,
                  //                 ^^^^ BUG HERE! ^^^^
                  realized_delta, equity_after, pos_after, chain_id);
```

**THE PROBLEM:**
- `std::abs(trade_qty)` always records **positive quantities**
- SELL orders should record **negative quantities** (-20 shares)
- But audit system receives **positive quantities** (+20 shares)
- This causes position accumulation instead of proper reduction

### **EVIDENCE:**
| **System** | **Trade Recording** | **Position Tracking** |
|------------|--------------------|-----------------------|
| **StratTest** | ✅ Correct: -20 shares | ✅ Correct: Position reduces |
| **Audit** | ❌ Wrong: +20 shares | ❌ Wrong: Position grows |

## 📋 IMMEDIATE ACTIONS

1. ✅ Run strattest with detailed final state logging
2. ⏳ Examine audit recording code (`event_fill_ex`)
3. ⏳ Compare strattest final state vs audit final state
4. ⏳ Add position tracking debug to identify discrepancy point

---

**PRIORITY**: CRITICAL - System integrity compromised
**IMPACT**: All trading strategies affected  
**STATUS**: Active investigation required


---

## 📋 **TABLE OF CONTENTS**

1. [temp_mega_doc/POSITION_ACCUMULATION_BUG_REPORT.md](#file-1)
2. [temp_mega_doc/audit.cpp](#file-2)
3. [temp_mega_doc/core.hpp](#file-3)
4. [temp_mega_doc/runner.cpp](#file-4)

---

## 📄 **FILE 1 of 4**: temp_mega_doc/POSITION_ACCUMULATION_BUG_REPORT.md

**File Information**:
- **Path**: `temp_mega_doc/POSITION_ACCUMULATION_BUG_REPORT.md`

- **Size**: 72 lines
- **Modified**: 2025-09-19 13:05:51

- **Type**: .md

```text
# CRITICAL BUG REPORT: Position Accumulation - Audit vs StratTest Discrepancy

## 🚨 EXECUTIVE SUMMARY

**CRITICAL FINDING**: After database reset, the audit system STILL shows massive position accumulation and conflicting positions. This indicates a real, current bug in either:
1. The audit recording system (recording wrong data)
2. The strattest execution system (executing wrong trades)
3. A disconnect between what strattest reports vs what actually executes

## 📊 EVIDENCE FROM FRESH DATABASE

### **Audit System Output (Run ID: 225376)**
```
│ Cash Balance        │ $-740108.71 │ Position Value      │ $ 840691.85 │
│ PSQ        │      599 │ $     80239.05 │
│ QQQ        │      432 │ $    253686.36 │
│ SQQQ       │     9338 │ $    169410.91 │
│ TQQQ       │      215 │ $    337355.52 │
│ Total Trades        │        3216 │
```

### **Trade Pattern Analysis**
```
│ TQQQ   │ 🟢BUY  │    0.327 │ → TQQQ:236 shares
│ TQQQ   │ 🔴SELL │    0.307 │ → TQQQ:235 shares  
│ TQQQ   │ 🟢BUY  │    0.294 │ → TQQQ:236 shares
```

**CRITICAL ISSUE**: Tiny trades (0.3 shares) somehow result in massive positions (236 shares)!

## 🔍 ROOT CAUSE IDENTIFIED!

### **✅ STRATTEST EXECUTION IS CORRECT**
**Debug Output Confirms:**
- Position updates work perfectly: `[APPLY_FILL] SID:0 | Final: 0`
- Final positions: QQQ: 0 shares, TQQQ: 0 shares (properly closed at EOD)
- Reasonable trade sizes: 55.0563 shares, 20.6575 shares
- Total fills: 304 (matches strattest report)

### **🚨 AUDIT RECORDING BUG FOUND**
**Line 195 in `src/runner.cpp`:**
```cpp
audit.event_fill_ex(bar.ts_utc_epoch, decision.instrument, 
                  instrument_price, std::abs(trade_qty), fees, side,
                  //                 ^^^^ BUG HERE! ^^^^
                  realized_delta, equity_after, pos_after, chain_id);
```

**THE PROBLEM:**
- `std::abs(trade_qty)` always records **positive quantities**
- SELL orders should record **negative quantities** (-20 shares)
- But audit system receives **positive quantities** (+20 shares)
- This causes position accumulation instead of proper reduction

### **EVIDENCE:**
| **System** | **Trade Recording** | **Position Tracking** |
|------------|--------------------|-----------------------|
| **StratTest** | ✅ Correct: -20 shares | ✅ Correct: Position reduces |
| **Audit** | ❌ Wrong: +20 shares | ❌ Wrong: Position grows |

## 📋 IMMEDIATE ACTIONS

1. ✅ Run strattest with detailed final state logging
2. ⏳ Examine audit recording code (`event_fill_ex`)
3. ⏳ Compare strattest final state vs audit final state
4. ⏳ Add position tracking debug to identify discrepancy point

---

**PRIORITY**: CRITICAL - System integrity compromised
**IMPACT**: All trading strategies affected  
**STATUS**: Active investigation required

```

## 📄 **FILE 2 of 4**: temp_mega_doc/audit.cpp

**File Information**:
- **Path**: `temp_mega_doc/audit.cpp`

- **Size**: 319 lines
- **Modified**: 2025-09-19 13:05:51

- **Type**: .cpp

```text
#include "sentio/audit.hpp"
#include "sentio/signal_diag.hpp"
#include <cstring>
#include <sstream>
#include <iomanip>

// ---- minimal SHA1 (tiny, not constant-time; for tamper detection only) ----
namespace {
struct Sha1 {
  uint32_t h0=0x67452301, h1=0xEFCDAB89, h2=0x98BADCFE, h3=0x10325476, h4=0xC3D2E1F0;
  std::string hex(const std::string& s){
    uint64_t ml = s.size()*8ULL;
    std::string msg = s;
    msg.push_back('\x80');
    while ((msg.size()%64)!=56) msg.push_back('\0');
    for (int i=7;i>=0;--i) msg.push_back(char((ml>>(i*8))&0xFF));
    for (size_t i=0;i<msg.size(); i+=64) {
      uint32_t w[80];
      for (int t=0;t<16;++t) {
        w[t] = (uint32_t(uint8_t(msg[i+4*t]))<<24)
             | (uint32_t(uint8_t(msg[i+4*t+1]))<<16)
             | (uint32_t(uint8_t(msg[i+4*t+2]))<<8)
             | (uint32_t(uint8_t(msg[i+4*t+3])));
      }
      for (int t=16;t<80;++t){ uint32_t v = w[t-3]^w[t-8]^w[t-14]^w[t-16]; w[t]=(v<<1)|(v>>31); }
      uint32_t a=h0,b=h1,c=h2,d=h3,e=h4;
      for (int t=0;t<80;++t){
        uint32_t f,k;
        if (t<20){ f=(b&c)|((~b)&d); k=0x5A827999; }
        else if (t<40){ f=b^c^d; k=0x6ED9EBA1; }
        else if (t<60){ f=(b&c)|(b&d)|(c&d); k=0x8F1BBCDC; }
        else { f=b^c^d; k=0xCA62C1D6; }
        uint32_t temp = ((a<<5)|(a>>27)) + f + e + k + w[t];
        e=d; d=c; c=((b<<30)|(b>>2)); b=a; a=temp;
      }
      h0+=a; h1+=b; h2+=c; h3+=d; h4+=e;
    }
    std::ostringstream os; os<<std::hex<<std::setfill('0')<<std::nouppercase;
    os<<std::setw(8)<<h0<<std::setw(8)<<h1<<std::setw(8)<<h2<<std::setw(8)<<h3<<std::setw(8)<<h4;
    return os.str();
  }
};
}

namespace sentio {

static inline std::string num_s(double v){
  std::ostringstream os; os.setf(std::ios::fixed); os<<std::setprecision(8)<<v; return os.str();
}
static inline std::string num_i(std::int64_t v){
  std::ostringstream os; os<<v; return os.str();
}
std::string AuditRecorder::json_escape_(const std::string& s){
  std::string o; o.reserve(s.size()+8);
  for (char c: s){
    switch(c){
      case '"': o+="\\\""; break;
      case '\\':o+="\\\\"; break;
      case '\b':o+="\\b"; break;
      case '\f':o+="\\f"; break;
      case '\n':o+="\\n"; break;
      case '\r':o+="\\r"; break;
      case '\t':o+="\\t"; break;
      default: o.push_back(c);
    }
  }
  return o;
}

std::string AuditRecorder::sha1_hex_(const std::string& s){
  Sha1 sh; return sh.hex(s);
}

AuditRecorder::AuditRecorder(const AuditConfig& cfg)
: run_id_(cfg.run_id), file_path_(cfg.file_path), flush_each_(cfg.flush_each)
{
  fp_ = std::fopen(cfg.file_path.c_str(), "ab");
  if (!fp_) throw std::runtime_error("Audit open failed: "+cfg.file_path);
}
AuditRecorder::~AuditRecorder(){ if (fp_) std::fclose(fp_); }

void AuditRecorder::write_line_(const std::string& body){
  std::string core = "{\"run\":\""+json_escape_(run_id_)+"\",\"seq\":"+num_i((std::int64_t)seq_)+","+body+"}";
  std::string line = core; // sha1 over core for stability
  std::string h = sha1_hex_(core);
  line.pop_back(); // remove trailing '}'
  line += ",\"sha1\":\""+h+"\"}\n";
  if (std::fwrite(line.data(),1,line.size(),fp_)!=line.size()) throw std::runtime_error("Audit write failed");
  if (flush_each_) std::fflush(fp_);
  ++seq_;
}

void AuditRecorder::event_run_start(std::int64_t ts, const std::string& meta){
  write_line_("\"type\":\"run_start\",\"ts\":"+num_i(ts)+",\"meta\":"+meta+"}");
}
void AuditRecorder::event_run_end(std::int64_t ts, const std::string& meta){
  write_line_("\"type\":\"run_end\",\"ts\":"+num_i(ts)+",\"meta\":"+meta+"}");
}
void AuditRecorder::event_bar(std::int64_t ts, const std::string& inst, double open, double high, double low, double close, double volume){
  write_line_("\"type\":\"bar\",\"ts\":"+num_i(ts)+",\"inst\":\""+json_escape_(inst)+"\",\"o\":"+num_s(open)+",\"h\":"+num_s(high)+",\"l\":"+num_s(low)+",\"c\":"+num_s(close)+",\"v\":"+num_s(volume)+"}");
}
void AuditRecorder::event_signal(std::int64_t ts, const std::string& base, SigType t, double conf){
  write_line_("\"type\":\"signal\",\"ts\":"+num_i(ts)+",\"base\":\""+json_escape_(base)+"\",\"sig\":" + num_i((int)t) + ",\"p\":"+num_s(conf)+"}");
}
void AuditRecorder::event_route (std::int64_t ts, const std::string& base, const std::string& inst, double tw){
  write_line_("\"type\":\"route\",\"ts\":"+num_i(ts)+",\"base\":\""+json_escape_(base)+"\",\"inst\":\""+json_escape_(inst)+"\",\"tw\":"+num_s(tw)+"}");
}
void AuditRecorder::event_order (std::int64_t ts, const std::string& inst, Side side, double qty, double limit_px){
  write_line_("\"type\":\"order\",\"ts\":"+num_i(ts)+",\"inst\":\""+json_escape_(inst)+"\",\"side\":"+num_i((int)side)+",\"qty\":"+num_s(qty)+",\"limit\":"+num_s(limit_px)+"}");
}
void AuditRecorder::event_fill  (std::int64_t ts, const std::string& inst, double price, double qty, double fees, Side side){
  write_line_("\"type\":\"fill\",\"ts\":"+num_i(ts)+",\"inst\":\""+json_escape_(inst)+"\",\"px\":"+num_s(price)+",\"qty\":"+num_s(qty)+",\"fees\":"+num_s(fees)+",\"side\":"+num_i((int)side)+"}");
}
void AuditRecorder::event_snapshot(std::int64_t ts, const AccountState& a){
  write_line_("\"type\":\"snapshot\",\"ts\":"+num_i(ts)+",\"cash\":"+num_s(a.cash)+",\"real\":"+num_s(a.realized)+",\"equity\":"+num_s(a.equity)+"}");
}
void AuditRecorder::event_metric(std::int64_t ts, const std::string& key, double val){
  write_line_("\"type\":\"metric\",\"ts\":"+num_i(ts)+",\"key\":\""+json_escape_(key)+"\",\"val\":"+num_s(val)+"}");
}

// ----------------- Extended events --------------------
void AuditRecorder::event_signal_ex(std::int64_t ts, const std::string& base, SigType t, double conf,
                                    const std::string& chain_id){
  write_line_("\"type\":\"signal\",\"ts\":"+num_i(ts)+",\"base\":\""+json_escape_(base)+"\",\"sig\":" + num_i((int)t) + ",\"p\":"+num_s(conf)+",\"chain\":\""+json_escape_(chain_id)+"\"}");
}
void AuditRecorder::event_route_ex (std::int64_t ts, const std::string& base, const std::string& inst, double tw,
                                    const std::string& chain_id){
  write_line_("\"type\":\"route\",\"ts\":"+num_i(ts)+",\"base\":\""+json_escape_(base)+"\",\"inst\":\""+json_escape_(inst)+"\",\"tw\":"+num_s(tw)+",\"chain\":\""+json_escape_(chain_id)+"\"}");
}
void AuditRecorder::event_order_ex (std::int64_t ts, const std::string& inst, Side side, double qty, double limit_px,
                                    const std::string& chain_id){
  write_line_("\"type\":\"order\",\"ts\":"+num_i(ts)+",\"inst\":\""+json_escape_(inst)+"\",\"side\":"+num_i((int)side)+",\"qty\":"+num_s(qty)+",\"limit\":"+num_s(limit_px)+",\"chain\":\""+json_escape_(chain_id)+"\"}");
}
void AuditRecorder::event_fill_ex  (std::int64_t ts, const std::string& inst, double price, double qty, double fees, Side side,
                                    double realized_pnl_delta, double equity_after, double position_after,
                                    const std::string& chain_id){
  write_line_(
    "\"type\":\"fill\",\"ts\":"+num_i(ts)+
    ",\"inst\":\""+json_escape_(inst)+
    "\",\"px\":"+num_s(price)+
    ",\"qty\":"+num_s(qty)+
    ",\"fees\":"+num_s(fees)+
    ",\"side\":"+num_i((int)side)+
    ",\"pnl_d\":"+num_s(realized_pnl_delta)+
    ",\"eq_after\":"+num_s(equity_after)+
    ",\"pos_after\":"+num_s(position_after)+
    ",\"chain\":\""+json_escape_(chain_id)+"\"}"
  );
}

// ----------------- Signal Diagnostics Events --------------------
void AuditRecorder::event_signal_diag(std::int64_t ts, const std::string& strategy_name, const SignalDiag& diag) {
  write_line_(
    "\"type\":\"signal_diag\",\"ts\":"+num_i(ts)+
    ",\"strategy\":\""+json_escape_(strategy_name)+"\""
    ",\"emitted\":"+num_i(diag.emitted)+
    ",\"dropped\":"+num_i(diag.dropped)+
    ",\"r_min_bars\":"+num_i(diag.r_min_bars)+
    ",\"r_session\":"+num_i(diag.r_session)+
    ",\"r_nan\":"+num_i(diag.r_nan)+
    ",\"r_zero_vol\":"+num_i(diag.r_zero_vol)+
    ",\"r_threshold\":"+num_i(diag.r_threshold)+
    ",\"r_cooldown\":"+num_i(diag.r_cooldown)+
    ",\"r_dup\":"+num_i(diag.r_dup)
  );
}

void AuditRecorder::event_signal_drop(std::int64_t ts, const std::string& strategy_name, const std::string& symbol, 
                                      DropReason reason, const std::string& chain_id, const std::string& note) {
  write_line_(
    "\"type\":\"signal_drop\",\"ts\":"+num_i(ts)+
    ",\"strategy\":\""+json_escape_(strategy_name)+"\""
    ",\"symbol\":\""+json_escape_(symbol)+"\""
    ",\"reason\":"+num_i((int)reason)+
    ",\"chain\":\""+json_escape_(chain_id)+"\""
    ",\"note\":\""+json_escape_(note)+"\""
  );
}

// ----------------- Replayer --------------------

static inline bool parse_kv(const std::string& s, const char* key, std::string& out) {
  auto kq = std::string("\"")+key+"\":";
  auto p = s.find(kq); if (p==std::string::npos) return false;
  p += kq.size();
  if (p>=s.size()) return false;
  if (s[p]=='"'){ // string
    auto e = s.find('"', p+1);
    if (e==std::string::npos) return false;
    out = s.substr(p+1, e-(p+1));
    return true;
  } else { // number or enum
    auto e = s.find_first_of(",}", p);
    out = s.substr(p, e-p);
    return true;
  }
}

std::optional<ReplayResult> AuditReplayer::replay_file(const std::string& path,
                                                       const std::string& run_expect)
{
  std::FILE* fp = std::fopen(path.c_str(), "rb");
  if (!fp) return std::nullopt;

  PriceBook pb;
  ReplayResult rr;
  rr.acct.cash = 0.0;
  rr.acct.realized = 0.0;

  char buf[16*1024];
  while (std::fgets(buf, sizeof(buf), fp)) {
    std::string line(buf);
    // very light JSONL parsing (we control writer)

    std::string run; if (!parse_kv(line, "run", run)) continue;
    if (!run_expect.empty() && run!=run_expect) continue;

    std::string type; if (!parse_kv(line, "type", type)) continue;
    // Note: timestamp parsing removed as it's not currently used in replay logic

    if (type=="bar") {
      std::string inst; parse_kv(line, "inst", inst);
      std::string o,h,l,c; parse_kv(line,"o",o); parse_kv(line,"h",h);
      parse_kv(line,"l",l); parse_kv(line,"c",c);
      AuditBar b{std::stod(o),std::stod(h),std::stod(l),std::stod(c)};
      apply_bar_(pb, inst, b);
      ++rr.bars;
    } else if (type=="signal") {
      ++rr.signals;
    } else if (type=="route") {
      ++rr.routes;
    } else if (type=="order") {
      ++rr.orders;
    } else if (type=="fill") {
      std::string inst,px,qty,fees,side_s;
      parse_kv(line,"inst",inst); parse_kv(line,"px",px);
      parse_kv(line,"qty",qty); parse_kv(line,"fees",fees);
      parse_kv(line,"side",side_s);
      Side side = side_s.empty() ? Side::Buy : static_cast<Side>(std::stoi(side_s));
      apply_fill_(rr, inst, std::stod(px), std::stod(qty), std::stod(fees), side);
      ++rr.fills;
      mark_to_market_(pb, rr);
    } else if (type=="snapshot") {
      // snapshots are optional for verification; we recompute anyway
      // you can cross-check here if you also store snapshots
    } else if (type=="run_end") {
      // could verify sha1 continuity/counts here
    }
  }
  std::fclose(fp);

  mark_to_market_(pb, rr);
  return rr;
}

bool AuditReplayer::apply_bar_(PriceBook& pb, const std::string& instrument, const AuditBar& b) {
  pb.last_px[instrument] = b.close;
  return true;
}

void AuditReplayer::apply_fill_(ReplayResult& rr, const std::string& inst, double px, double qty, double fees, Side side) {
  auto& pos = rr.positions[inst];
  
  // Convert order side to position impact
  // Buy orders: positive position qty, Sell orders: negative position qty
  double position_qty = (side == Side::Buy) ? qty : -qty;
  
  // cash impact: buy qty>0 => cash decreases, sell qty>0 => cash increases
  double cash_delta = (side == Side::Buy) ? -(px*qty + fees) : (px*qty - fees);
  rr.acct.cash += cash_delta;

  // position update (VWAP)
  double new_qty = pos.qty + position_qty;
  
  if (new_qty == 0.0) {
    // flat: realize P&L for the round trip
    if (pos.qty != 0.0) {
      rr.acct.realized += (px - pos.avg_px) * pos.qty;
    }
    pos.qty = 0.0; 
    pos.avg_px = 0.0;
  } else if (pos.qty == 0.0) {
    // opening new position
    pos.qty = new_qty;
    pos.avg_px = px;
  } else if ((pos.qty > 0) == (position_qty > 0)) {
    // adding to same side -> new average
    pos.avg_px = (pos.avg_px * pos.qty + px * position_qty) / new_qty;
    pos.qty = new_qty;
  } else {
    // reducing or flipping side -> realize partial P&L
    double closed_qty = std::min(std::abs(position_qty), std::abs(pos.qty));
    if (pos.qty > 0) {
      rr.acct.realized += (px - pos.avg_px) * closed_qty;
    } else {
      rr.acct.realized += (pos.avg_px - px) * closed_qty;
    }
    pos.qty = new_qty;
    if (pos.qty == 0.0) {
      pos.avg_px = 0.0;
    } else {
      pos.avg_px = px; // new average for remaining position
    }
  }
}

void AuditReplayer::mark_to_market_(const PriceBook& pb, ReplayResult& rr) {
  double mtm=0.0;
  for (auto& kv : rr.positions) {
    const auto& inst = kv.first;
    const auto& p = kv.second;
    auto it = pb.last_px.find(inst);
    if (it==pb.last_px.end()) continue;
    mtm += p.qty * it->second;
  }
  rr.acct.equity = rr.acct.cash + rr.acct.realized + mtm;
}

} // namespace sentio
```

## 📄 **FILE 3 of 4**: temp_mega_doc/core.hpp

**File Information**:
- **Path**: `temp_mega_doc/core.hpp`

- **Size**: 74 lines
- **Modified**: 2025-09-19 13:05:51

- **Type**: .hpp

```text
#pragma once
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cmath> // For std::sqrt

namespace sentio {

// Forward declarations
struct SymbolTable;

struct Bar {
    std::string ts_utc;
    int64_t ts_utc_epoch;
    double open, high, low, close;
    uint64_t volume;
};

// **MODIFIED**: This struct now holds a vector of Positions, indexed by symbol ID for performance.
struct Position { 
    double qty = 0.0; 
    double avg_price = 0.0; 
};

struct Portfolio {
    double cash = 100000.0;
    std::vector<Position> positions; // Indexed by symbol ID

    Portfolio() = default;
    explicit Portfolio(size_t num_symbols) : positions(num_symbols) {}
};

// **MODIFIED**: Vector-based functions are now the primary way to manage the portfolio.
inline void apply_fill(Portfolio& pf, int sid, double qty_delta, double price) {
    if (sid < 0 || static_cast<size_t>(sid) >= pf.positions.size()) {
        return; // Invalid symbol ID
    }
    
    pf.cash -= qty_delta * price;
    auto& pos = pf.positions[sid];
    
    double new_qty = pos.qty + qty_delta;
    if (std::abs(new_qty) < 1e-9) { // Position is closed
        pos.qty = 0.0;
        pos.avg_price = 0.0;
    } else {
        if (pos.qty * new_qty >= 0) { // Increasing position or opening a new one
            pos.avg_price = (pos.avg_price * pos.qty + price * qty_delta) / new_qty;
        }
        // If flipping from long to short or vice-versa, the new avg_price is just the fill price.
        else if (pos.qty * qty_delta < 0) {
             pos.avg_price = price;
        }
        pos.qty = new_qty;
    }
}

inline double equity_mark_to_market(const Portfolio& pf, const std::vector<double>& last_prices) {
    double eq = pf.cash;
    for (size_t sid = 0; sid < pf.positions.size(); ++sid) {
        if (std::abs(pf.positions[sid].qty) > 0.0 && sid < last_prices.size()) {
            eq += pf.positions[sid].qty * last_prices[sid];
        }
    }
    return eq;
}


// **REMOVED**: Old, simplistic Direction and StratSignal types are now deprecated.

} // namespace sentio
```

## 📄 **FILE 4 of 4**: temp_mega_doc/runner.cpp

**File Information**:
- **Path**: `temp_mega_doc/runner.cpp`

- **Size**: 1007 lines
- **Modified**: 2025-09-19 13:05:51

- **Type**: .cpp

```text
#include "sentio/runner.hpp"
#include "sentio/base_strategy.hpp"
#include "sentio/pricebook.hpp"
#include "sentio/metrics.hpp"
#include "audit/audit_db_recorder.hpp"
#include "sentio/safe_sizer.hpp"
#include "sentio/audit.hpp"
// Strategy-Agnostic Backend Components
#include "sentio/strategy_profiler.hpp"
#include "sentio/adaptive_allocation_manager.hpp"
#include "sentio/universal_position_coordinator.hpp"
#include "sentio/adaptive_eod_manager.hpp"
#include "sentio/feature_feeder.hpp"
#include "sentio/canonical_evaluation.hpp"
// Golden Rule Enforcement Components
#include "sentio/execution_verifier.hpp"
#include "sentio/circuit_breaker.hpp"
#include <iostream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <chrono>
#include <ctime>
#include <sqlite3.h>

namespace sentio {

// **GOLDEN RULE ENFORCEMENT PIPELINE**
// Strict phase ordering with enforcement mechanisms that cannot be bypassed
static void execute_bar_pipeline(
    double strategy_probability,
    Portfolio& portfolio, 
    const SymbolTable& ST, 
    const Pricebook& pricebook,
    StrategyProfiler& profiler,
    AdaptiveAllocationManager& allocation_mgr,
    UniversalPositionCoordinator& position_coord, 
    AdaptiveEODManager& eod_mgr,
    const std::vector<std::vector<Bar>>& series, 
    const Bar& bar,
    const std::string& chain_id,
    IAuditRecorder& audit,
    bool logging_enabled,
    int& total_fills,
    const std::string& strategy_name,
    size_t bar_index) {
    
    // ENFORCEMENT COMPONENTS - Cannot be bypassed
    static ExecutionVerifier verifier;
    static CircuitBreaker breaker;
    
    // PHASE 0: Reset and prepare for new bar
    verifier.reset_bar(bar.ts_utc_epoch);
    position_coord.reset_bar(bar.ts_utc_epoch);
    
    auto profile = profiler.get_current_profile();
    
    // PHASE 1: SKIP EOD CHECK (EOD requirement removed per user request)
    verifier.mark_eod_checked(bar.ts_utc_epoch);
    
    // PHASE 2: CIRCUIT BREAKER CHECK (Emergency protection)
    if (!breaker.check_portfolio_integrity(portfolio, ST, bar.ts_utc_epoch)) {
        if (breaker.is_tripped()) {
            
            auto emergency_orders = breaker.get_emergency_closure(portfolio, ST);
            if (!emergency_orders.empty() && verifier.verify_can_execute(bar.ts_utc_epoch, emergency_orders[0].instrument)) {
                const auto& decision = emergency_orders[0];
                
                // Execute emergency close
                int instrument_id = ST.get_id(decision.instrument);
                if (instrument_id != -1) {
                    portfolio.positions[instrument_id].qty = 0.0; // Force close
                    total_fills++;
                    verifier.mark_trade_executed(bar.ts_utc_epoch, decision.instrument);
                }
            }
            
            return; // GOLDEN RULE: No strategy consultation during emergency
        }
    }
    
    // PHASE 3: STRATEGY CONSULTATION (Only if phases 1 & 2 allow)
    if (!verifier.verify_can_execute(bar.ts_utc_epoch)) {
        return;
    }
    
    // Profile the strategy signal
    profiler.observe_signal(strategy_probability, bar.ts_utc_epoch);
    
    // Get adaptive allocations based on profile
    auto allocations = allocation_mgr.get_allocations(strategy_probability, profile);
    
    // Log signal activity
    if (logging_enabled && !allocations.empty()) {
        audit.event_signal_ex(bar.ts_utc_epoch, ST.get_symbol(0), 
                            SigType::BUY, strategy_probability, chain_id);
    }
    
    // Position coordination with enforcement
    verifier.mark_position_coordinated(bar.ts_utc_epoch);
    auto coordination_decisions = position_coord.coordinate(
        allocations, portfolio, ST, bar.ts_utc_epoch, profile
    );
    
    // PHASE 4: EXECUTE APPROVED DECISIONS (With strict enforcement)
    
    for (const auto& coord_decision : coordination_decisions) {
        if (coord_decision.result != CoordinationResult::APPROVED) {
            if (logging_enabled) {
                audit.event_signal_drop(bar.ts_utc_epoch, strategy_name, 
                                      coord_decision.decision.instrument,
                                      DropReason::THRESHOLD, chain_id, 
                                      coord_decision.reason);
            }
            continue;
        }
        
        const auto& decision = coord_decision.decision;
        
        // ENFORCEMENT: Verify execution is allowed
        bool is_closing_trade = (decision.target_weight == 0.0);
        if (!verifier.verify_can_execute(bar.ts_utc_epoch, decision.instrument, is_closing_trade)) {
            continue;
        }
        
        // Use SafeSizer for execution
        SafeSizer sizer;
        double target_qty = sizer.calculate_target_quantity(
            portfolio, ST, pricebook.last_px, 
            decision.instrument, decision.target_weight, 
            bar.ts_utc_epoch, series[ST.get_id(decision.instrument)]
        );
        
        int instrument_id = ST.get_id(decision.instrument);
        if (instrument_id == -1) continue;
        
        double current_qty = portfolio.positions[instrument_id].qty;
        double trade_qty = target_qty - current_qty;
        
        if (std::abs(trade_qty) < 1e-9) continue;
        
        double instrument_price = pricebook.last_px[instrument_id];
        if (instrument_price <= 0) continue;
        
        // Execute trade
        Side side = (trade_qty > 0) ? Side::Buy : Side::Sell;
        
        if (logging_enabled) {
            audit.event_order_ex(bar.ts_utc_epoch, decision.instrument, side, 
                               std::abs(trade_qty), 0.0, chain_id);
        }
        
        // Calculate P&L
        double realized_delta = 0.0;
        const auto& pos_before = portfolio.positions[instrument_id];
        double closing = 0.0;
        if (pos_before.qty > 0 && trade_qty < 0) {
            closing = std::min(std::abs(trade_qty), pos_before.qty);
        } else if (pos_before.qty < 0 && trade_qty > 0) {
            closing = std::min(std::abs(trade_qty), std::abs(pos_before.qty));
        }
        if (closing > 0.0) {
            if (pos_before.qty > 0) {
                realized_delta = (instrument_price - pos_before.avg_price) * closing;
            } else {
                realized_delta = (pos_before.avg_price - instrument_price) * closing;
            }
        }
        
        // Apply fill
        double fees = AlpacaCostModel::calculate_fees(
            decision.instrument, std::abs(trade_qty), instrument_price, side == Side::Sell
        );
        
        apply_fill(portfolio, instrument_id, trade_qty, instrument_price);
        portfolio.cash -= fees;
        
        // **CRITICAL SAFETY CHECK**: Prevent negative cash balance
        if (portfolio.cash < 0) {
            std::cerr << "ERROR: Negative cash balance detected after trade: " 
                      << portfolio.cash << " (instrument: " << decision.instrument 
                      << ", qty: " << trade_qty << ", price: " << instrument_price << ")" << std::endl;
            // Emergency reversal: undo the trade
            apply_fill(portfolio, instrument_id, -trade_qty, instrument_price);
            portfolio.cash += fees;
            std::cerr << "Trade reversed. Cash restored to: " << portfolio.cash << std::endl;
            continue; // Skip this trade
        }
        
        double equity_after = equity_mark_to_market(portfolio, pricebook.last_px);
        double pos_after = portfolio.positions[instrument_id].qty;
        
        if (logging_enabled) {
            // **CRITICAL FIX**: Use signed trade_qty, not std::abs(trade_qty)
            // SELL orders need negative quantities for proper position tracking
            audit.event_fill_ex(bar.ts_utc_epoch, decision.instrument, 
                              instrument_price, trade_qty, fees, side,
                              realized_delta, equity_after, pos_after, chain_id);
        }
        
        // Update profiler with trade observation
        profiler.observe_trade(strategy_probability, decision.instrument, bar.ts_utc_epoch);
        total_fills++;
        
        // ENFORCEMENT: Mark trade as executed
        verifier.mark_trade_executed(bar.ts_utc_epoch, decision.instrument, is_closing_trade);
    }
}


// CHANGED: The function now returns a BacktestOutput struct with raw data and accepts dataset metadata.
BacktestOutput run_backtest(IAuditRecorder& audit, const SymbolTable& ST, const std::vector<std::vector<Bar>>& series, 
                      int base_symbol_id, const RunnerCfg& cfg, const DatasetMetadata& dataset_meta, 
                      StrategyProfiler* persistent_profiler) {
    
    // 1. ============== INITIALIZATION ==============
    BacktestOutput output{}; // NEW: Initialize the output struct
    
    const bool logging_enabled = (cfg.audit_level == AuditLevel::Full);
    // Calculate actual test period in trading days
    int actual_test_days = 0;
    if (!series[base_symbol_id].empty()) {
        // Estimate trading days from bars (assuming ~390 bars per trading day)
        actual_test_days = std::max(1, static_cast<int>(series[base_symbol_id].size()) / 390);
    }
    
    // Determine dataset type based on data source and characteristics
    std::string dataset_type = "historical"; // default
    if (!series[base_symbol_id].empty()) {
        // Check if this looks like future/AI regime data
        // Future data characteristics: ~26k bars (4 weeks), specific timestamp patterns
        size_t bar_count = series[base_symbol_id].size();
        std::int64_t first_ts = series[base_symbol_id][0].ts_utc_epoch;
        std::int64_t last_ts = series[base_symbol_id].back().ts_utc_epoch;
        double time_span_days = (last_ts - first_ts) / (60.0 * 60.0 * 24.0); // Convert seconds to days
        
        // Future data is typically exactly 4 weeks (28 days) with ~26k bars
        if (bar_count >= 25000 && bar_count <= 27000 && time_span_days >= 27 && time_span_days <= 29) {
            dataset_type = "future_ai_regime";
        }
        // Historical data is typically longer periods or different bar counts
    }
    
    // **DEFERRED**: Calculate actual test period metadata after we know the filtered data range
    // This will be done after warmup calculation when we know the exact bars being processed
    
    auto strategy = StrategyFactory::instance().create_strategy(cfg.strategy_name);
    if (!strategy) {
        std::cerr << "FATAL: Could not create strategy '" << cfg.strategy_name << "'. Check registration." << std::endl;
        return output;
    }
    
    ParameterMap params;
    for (const auto& [key, value] : cfg.strategy_params) {
        try {
            params[key] = std::stod(value);
        } catch (...) { /* ignore */ }
    }
    strategy->set_params(params);

    Portfolio portfolio(ST.size());
    Pricebook pricebook(base_symbol_id, ST, series);
    
    // **STRATEGY-AGNOSTIC EXECUTION PIPELINE COMPONENTS**
    StrategyProfiler local_profiler;
    StrategyProfiler& profiler = persistent_profiler ? *persistent_profiler : local_profiler;
    AdaptiveAllocationManager adaptive_allocation_mgr;
    UniversalPositionCoordinator universal_position_coord;
    AdaptiveEODManager adaptive_eod_mgr;
    
    std::vector<std::pair<std::string, double>> equity_curve;
    std::vector<std::int64_t> equity_curve_ts_ms;
    const auto& base_series = series[base_symbol_id];
    equity_curve.reserve(base_series.size());

    int total_fills = 0;
    int no_route_count = 0;
    int no_qty_count = 0;
    double cumulative_realized_pnl = 0.0;  // Track cumulative realized P&L for audit transparency

    // 2. ============== MAIN EVENT LOOP ==============
    size_t total_bars = base_series.size();
    size_t progress_interval = total_bars / 20; // 5% intervals (20 steps)
    
    // Skip first 300 bars to allow technical indicators to warm up
    size_t warmup_bars = 300;
    if (total_bars <= warmup_bars) {
        std::cout << "Warning: Not enough bars for warmup (need " << warmup_bars << ", have " << total_bars << ")" << std::endl;
        warmup_bars = 0;
    }
    
    // **CANONICAL METADATA**: Calculate actual test period from filtered data (post-warmup)
    std::int64_t run_period_start_ts_ms = 0;
    std::int64_t run_period_end_ts_ms = 0;
    int run_trading_days = 0;
    
    if (warmup_bars < base_series.size()) {
        run_period_start_ts_ms = base_series[warmup_bars].ts_utc_epoch * 1000;
        run_period_end_ts_ms = base_series.back().ts_utc_epoch * 1000;
        
        // Count unique trading days in the filtered range
        std::vector<std::int64_t> filtered_timestamps;
        for (size_t i = warmup_bars; i < base_series.size(); ++i) {
            filtered_timestamps.push_back(base_series[i].ts_utc_epoch * 1000);
        }
        run_trading_days = filtered_timestamps.size() / 390.0; // Approximate: 390 bars per trading day
    }
    
    // Start audit run with canonical metadata including dataset information
    std::string meta = "{";
    meta += "\"strategy\":\"" + cfg.strategy_name + "\",";
    meta += "\"base_symbol_id\":" + std::to_string(base_symbol_id) + ",";
    meta += "\"total_series\":" + std::to_string(series.size()) + ",";
    meta += "\"base_series_size\":" + std::to_string(series[base_symbol_id].size()) + ",";
    meta += "\"dataset_type\":\"" + dataset_type + "\",";
    meta += "\"test_period_days\":" + std::to_string(run_trading_days) + ",";
    meta += "\"run_period_start_ts_ms\":" + std::to_string(run_period_start_ts_ms) + ",";
    meta += "\"run_period_end_ts_ms\":" + std::to_string(run_period_end_ts_ms) + ",";
    meta += "\"run_trading_days\":" + std::to_string(run_trading_days) + ",";
    // **DATASET TRACEABILITY**: Include comprehensive dataset metadata
    meta += "\"dataset_source_type\":\"" + (dataset_meta.source_type.empty() ? dataset_type : dataset_meta.source_type) + "\",";
    meta += "\"dataset_file_path\":\"" + dataset_meta.file_path + "\",";
    meta += "\"dataset_file_hash\":\"" + dataset_meta.file_hash + "\",";
    meta += "\"dataset_track_id\":\"" + dataset_meta.track_id + "\",";
    meta += "\"dataset_regime\":\"" + dataset_meta.regime + "\",";
    meta += "\"dataset_bars_count\":" + std::to_string(dataset_meta.bars_count > 0 ? dataset_meta.bars_count : static_cast<int>(series[base_symbol_id].size())) + ",";
    meta += "\"dataset_time_range_start\":" + std::to_string(dataset_meta.time_range_start > 0 ? dataset_meta.time_range_start : run_period_start_ts_ms) + ",";
    meta += "\"dataset_time_range_end\":" + std::to_string(dataset_meta.time_range_end > 0 ? dataset_meta.time_range_end : run_period_end_ts_ms);
    meta += "}";
    
    // Use current time for run timestamp (for proper run ordering)
    std::int64_t start_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    if (logging_enabled && !cfg.skip_audit_run_creation) audit.event_run_start(start_ts, meta);
    
    for (size_t i = warmup_bars; i < base_series.size(); ++i) {
        
        const auto& bar = base_series[i];
        
        
        // **RENOVATED**: Governor handles day trading automatically - no manual time logic needed
        pricebook.sync_to_base_i(i);
        
        // Log bar data
        AuditBar audit_bar{bar.open, bar.high, bar.low, bar.close, static_cast<double>(bar.volume)};
        if (logging_enabled) audit.event_bar(bar.ts_utc_epoch, ST.get_symbol(base_symbol_id), audit_bar.open, audit_bar.high, audit_bar.low, audit_bar.close, audit_bar.volume);
        
        // **STRATEGY-AGNOSTIC**: Feed features to any strategy that needs them
        FeatureFeeder::feed_features_to_strategy(strategy.get(), base_series, i, strategy->get_name());
        
        // **RENOVATED ARCHITECTURE**: Governor-based target weight system
        
        // **STRATEGY-AGNOSTIC ARCHITECTURE**: Let strategy control its execution path
        std::string chain_id = std::to_string(bar.ts_utc_epoch) + ":" + std::to_string((long long)i);
        
        // Get strategy probability for logging
        double probability = strategy->calculate_probability(base_series, i);
        std::string base_symbol = ST.get_symbol(base_symbol_id);
        
        // **STRATEGY-AGNOSTIC EXECUTION PIPELINE**: Adaptive components that work for any strategy
        // Execute the Golden Rule pipeline with strategy probability
        execute_bar_pipeline(
            probability, portfolio, ST, pricebook,
            profiler, adaptive_allocation_mgr, universal_position_coord, adaptive_eod_mgr,
            series, bar, chain_id, audit, logging_enabled, total_fills, cfg.strategy_name, i
        );
        
        // Audit logging configured based on system settings
        
        // **STRATEGY-AGNOSTIC**: Log signal for diagnostics
        if (logging_enabled) {
            std::string signal_desc = strategy->get_signal_description(probability);
            
            // **STRATEGY-AGNOSTIC**: Convert signal description to SigType enum
            SigType sig_type = SigType::HOLD;
            std::string upper_desc = signal_desc;
            std::transform(upper_desc.begin(), upper_desc.end(), upper_desc.begin(), ::toupper);
            
            if (upper_desc.find("STRONG") != std::string::npos && upper_desc.find("BUY") != std::string::npos) {
                sig_type = SigType::STRONG_BUY;
            } else if (upper_desc.find("STRONG") != std::string::npos && upper_desc.find("SELL") != std::string::npos) {
                sig_type = SigType::STRONG_SELL;
            } else if (upper_desc.find("BUY") != std::string::npos) {
                sig_type = SigType::BUY;
            } else if (upper_desc.find("SELL") != std::string::npos) {
                sig_type = SigType::SELL;
            }
            // Default remains SigType::HOLD for any other signal descriptions
            
            audit.event_signal_ex(bar.ts_utc_epoch, ST.get_symbol(base_symbol_id), 
                                sig_type, probability, chain_id);
        }
        
        
        // 3. ============== SNAPSHOT ==============
        if (i % cfg.snapshot_stride == 0 || i == base_series.size() - 1) {
            double current_equity = equity_mark_to_market(portfolio, pricebook.last_px);
            
            // Fix: Ensure we have a valid timestamp string for metrics calculation
            std::string timestamp = bar.ts_utc;
            if (timestamp.empty()) {
                // Create synthetic progressive timestamps for metrics calculation
                // Start from a base date and add minutes for each bar
                static time_t base_time = 1726200000; // Sept 13, 2024 (recent date)
                time_t synthetic_time = base_time + (i * 60); // Add 1 minute per bar
                
                auto tm_val = *std::gmtime(&synthetic_time);
                char buffer[32];
                std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm_val);
                timestamp = std::string(buffer);
            }
            
            equity_curve.emplace_back(timestamp, current_equity);
            equity_curve_ts_ms.emplace_back(static_cast<std::int64_t>(bar.ts_utc_epoch) * 1000);
            
            // Log account snapshot
            // Calculate actual position value and track cumulative realized P&L  
            // double position_value = current_equity - portfolio.cash; // Unused during cleanup
            
            AccountState state;
            state.cash = portfolio.cash;
            state.equity = current_equity;
            state.realized = cumulative_realized_pnl; // Track actual cumulative realized P&L
            if (logging_enabled) audit.event_snapshot(bar.ts_utc_epoch, state);
        }
    }
    
    // 4. ============== METRICS & DIAGNOSTICS ==============
    strategy->get_diag().print(strategy->get_name().c_str());
    
    // Log signal diagnostics to audit trail
    if (logging_enabled) {
        audit.event_signal_diag(series[base_symbol_id].back().ts_utc_epoch, 
                               cfg.strategy_name, strategy->get_diag());
    }

    if (equity_curve.empty()) {
        return output;
    }
    
    // 3. ============== RAW DATA COLLECTION COMPLETE ==============
    // All metric calculation logic moved to UnifiedMetricsCalculator

    // 4. ============== POPULATE OUTPUT & RETURN ==============
    
    // NEW: Populate the output struct with the raw data from the simulation.
    output.equity_curve = equity_curve;
    output.equity_curve_ts_ms = equity_curve_ts_ms;
    output.total_fills = total_fills;
    output.no_route_events = no_route_count;
    output.no_qty_events = no_qty_count;
    output.run_trading_days = run_trading_days;

    // Audit system reconstructs equity curve for metrics

    // Log strategy profile analysis
    auto final_profile = profiler.get_current_profile();
    std::cout << "\n📊 Strategy Profile Analysis:\n";
    std::cout << "  Trading Style: " << 
        (final_profile.style == TradingStyle::AGGRESSIVE ? "AGGRESSIVE" :
         final_profile.style == TradingStyle::CONSERVATIVE ? "CONSERVATIVE" :
         final_profile.style == TradingStyle::BURST ? "BURST" : "ADAPTIVE") << "\n";
    std::cout << "  Avg Signal Frequency: " << std::fixed << std::setprecision(3) << final_profile.avg_signal_frequency << "\n";
    std::cout << "  Signal Volatility: " << std::fixed << std::setprecision(3) << final_profile.signal_volatility << "\n";
    std::cout << "  Trades per Block: " << std::fixed << std::setprecision(1) << final_profile.trades_per_block << "\n";
    std::cout << "  Adaptive Thresholds: 1x=" << std::fixed << std::setprecision(2) << final_profile.adaptive_entry_1x 
              << ", 3x=" << final_profile.adaptive_entry_3x << "\n";
    std::cout << "  Profile Confidence: " << std::fixed << std::setprecision(1) << (final_profile.confidence_level * 100) << "%\n";

    // Log the end of the run to the audit trail
    std::string end_meta = "{}";
    if (logging_enabled) {
        std::int64_t end_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        audit.event_run_end(end_ts, end_meta);
    }

    return output;
}

// ============== CANONICAL EVALUATION SYSTEM ==============

CanonicalReport run_canonical_backtest(
    IAuditRecorder& audit, 
    const SymbolTable& ST, 
    const std::vector<std::vector<Bar>>& series, 
    int base_symbol_id, 
    const RunnerCfg& cfg, 
    const DatasetMetadata& dataset_meta,
    const TradingBlockConfig& block_config) {
    
    // Create the main audit run for the canonical backtest
    std::string meta = "{";
    meta += "\"strategy\":\"" + cfg.strategy_name + "\",";
    meta += "\"base_symbol_id\":" + std::to_string(base_symbol_id) + ",";
    meta += "\"trading_blocks\":" + std::to_string(block_config.num_blocks) + ",";
    meta += "\"block_size\":" + std::to_string(block_config.block_size) + ",";
    meta += "\"dataset_source_type\":\"" + dataset_meta.source_type + "\",";
    meta += "\"dataset_file_path\":\"" + dataset_meta.file_path + "\",";
    meta += "\"dataset_regime\":\"" + dataset_meta.regime + "\",";
    meta += "\"evaluation_type\":\"canonical_trading_blocks\"";
    meta += "}";
    
    std::int64_t start_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    audit.event_run_start(start_ts, meta);
    
    CanonicalReport report;
    report.config = block_config;
    report.strategy_name = cfg.strategy_name;
    report.dataset_source = dataset_meta.source_type;
    
    const auto& base_series = series[base_symbol_id];
    
    // Calculate warmup bars - proportional to new 480-bar Trading Blocks
    // Use ~250 bars warmup (about half a Trading Block) for technical indicators
    size_t warmup_bars = 250;
    if (base_series.size() <= warmup_bars) {
        std::cout << "Warning: Not enough bars for warmup (need " << warmup_bars << ", have " << base_series.size() << ")" << std::endl;
        warmup_bars = 0;
    }
    
    // Calculate total bars needed for the canonical test
    size_t total_bars_needed = warmup_bars + block_config.total_bars();
    if (base_series.size() < total_bars_needed) {
        std::cout << "Warning: Not enough data for complete test (need " << total_bars_needed 
                  << ", have " << base_series.size() << "). Running partial test." << std::endl;
        // Adjust block count to fit available data
        size_t available_test_bars = base_series.size() - warmup_bars;
        int possible_blocks = static_cast<int>(available_test_bars / block_config.block_size);
        if (possible_blocks == 0) {
            std::cerr << "Error: Not enough data for even one block" << std::endl;
            return report;
        }
        // Note: We'll process only the possible blocks
    }
    
    // Calculate test period using most recent data (work backwards from end)
    size_t test_end_idx = base_series.size() - 1;
    size_t test_start_idx = test_end_idx - block_config.total_bars() + 1;
    size_t warmup_start_idx = test_start_idx - warmup_bars;
    
    // Store test period metadata
    report.test_start_ts_ms = base_series[test_start_idx].ts_utc_epoch * 1000;
    report.test_end_ts_ms = base_series[test_end_idx].ts_utc_epoch * 1000;
    
    std::vector<BlockResult> block_results;
    
    // **STRATEGY-AGNOSTIC**: Create persistent profiler across all blocks
    // This allows the profiler to learn the strategy's behavior over time
    StrategyProfiler persistent_profiler;
    
    // Process each block (using most recent data)
    for (int block_index = 0; block_index < block_config.num_blocks; ++block_index) {
        size_t block_start_idx = test_start_idx + (block_index * block_config.block_size);
        size_t block_end_idx = block_start_idx + block_config.block_size;
        
        // Check if we have enough data for this block
        if (block_end_idx > base_series.size()) {
            std::cout << "Insufficient data for block " << block_index << ". Stopping at " 
                      << block_results.size() << " completed blocks." << std::endl;
            break;
        }
        
        std::cout << "Processing Trading Block " << (block_index + 1) << "/" << block_config.num_blocks 
                  << " (bars " << block_start_idx << "-" << (block_end_idx - 1) << ")..." << std::endl;
        
        // Create a data slice for this block (including warmup from the correct position)
        std::vector<std::vector<Bar>> block_series;
        block_series.reserve(series.size());
        
        // Calculate the actual warmup start for this block
        size_t block_warmup_start = (block_start_idx >= warmup_bars) ? block_start_idx - warmup_bars : 0;
        
        for (const auto& symbol_series : series) {
            if (symbol_series.size() > block_end_idx) {
                // Include warmup + this block's data (from warmup start to block end)
                std::vector<Bar> slice(symbol_series.begin() + block_warmup_start, symbol_series.begin() + block_end_idx);
                block_series.push_back(slice);
            } else if (symbol_series.size() > block_start_idx) {
                // Partial data case
                std::vector<Bar> slice(symbol_series.begin() + block_warmup_start, symbol_series.end());
                block_series.push_back(slice);
            } else {
                // Empty series for this symbol in this block
                block_series.emplace_back();
            }
        }
        
        // Create block-specific dataset metadata
        DatasetMetadata block_meta = dataset_meta;
        if (!base_series.empty()) {
            block_meta.time_range_start = base_series[block_start_idx].ts_utc_epoch * 1000;
            block_meta.time_range_end = base_series[block_end_idx - 1].ts_utc_epoch * 1000;
            block_meta.bars_count = block_config.block_size;
        }
        
        // Get starting equity for this block
        double starting_equity = 100000.0; // Default starting capital
        if (!block_results.empty()) {
            starting_equity = block_results.back().ending_equity;
        }
        
        // Create block-specific config that skips audit run creation
        RunnerCfg block_cfg = cfg;
        block_cfg.skip_audit_run_creation = true;  // Skip audit run creation for individual blocks
        // ENSURE audit logging is enabled for instrument distribution
        block_cfg.audit_level = AuditLevel::Full;
        
        // Run backtest for this block with persistent profiler
        BacktestOutput block_output = run_backtest(audit, ST, block_series, base_symbol_id, block_cfg, block_meta, &persistent_profiler);
        
        // Calculate block metrics
        BlockResult block_result = CanonicalEvaluator::calculate_block_metrics(
            block_output.equity_curve,
            block_index,
            starting_equity,
            block_output.total_fills,
            base_series[block_start_idx].ts_utc_epoch * 1000,
            base_series[block_end_idx - 1].ts_utc_epoch * 1000
        );
        
        block_results.push_back(block_result);
        
        // **STRATEGY-AGNOSTIC**: Update profiler with block completion
        persistent_profiler.observe_block_complete(block_output.total_fills);
        
        std::cout << "Block " << (block_index + 1) << " completed: "
                  << "RPB=" << std::fixed << std::setprecision(4) << (block_result.return_per_block * 100) << "%, "
                  << "Sharpe=" << std::fixed << std::setprecision(2) << block_result.sharpe_ratio << ", "
                  << "Fills=" << block_result.fills << std::endl;
    }
    
    if (block_results.empty()) {
        std::cerr << "Error: No blocks were processed successfully" << std::endl;
        return report;
    }
    
    // Aggregate all block results
    report = CanonicalEvaluator::aggregate_block_results(block_config, block_results, cfg.strategy_name, dataset_meta.source_type);
    
    // Store block results in audit database (if it supports it)
    try {
        if (auto* db_recorder = dynamic_cast<audit::AuditDBRecorder*>(&audit)) {
            db_recorder->get_db().store_block_results(db_recorder->get_run_id(), block_results);
        }
    } catch (const std::exception& e) {
        std::cout << "Warning: Could not store block results in audit database: " << e.what() << std::endl;
    }
    
    // Helper function to convert timestamp to ISO format
    auto to_iso_string = [](std::int64_t timestamp_ms) -> std::string {
        std::time_t time_sec = timestamp_ms / 1000;
        std::tm* utc_tm = std::gmtime(&time_sec);
        
        char buffer[32];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", utc_tm);
        return std::string(buffer) + "Z";
    };
    
    // Get run ID from audit recorder
    std::string run_id = "unknown";
    if (auto* db_recorder = dynamic_cast<audit::AuditDBRecorder*>(&audit)) {
        run_id = db_recorder->get_run_id();
    }
    
    // Calculate trades per TB
    double trades_per_tb = 0.0;
    if (report.successful_blocks() > 0) {
        trades_per_tb = static_cast<double>(report.total_fills) / report.successful_blocks();
    }
    
    // Calculate MRB (Monthly Return per Block) - projected monthly return
    // Assuming ~20 Trading Blocks per month (480 bars/block, ~390 bars/day, ~20 trading days/month)
    double blocks_per_month = 20.0;
    double mrb = 0.0;
    if (report.mean_rpb != 0.0) {
        // Use compound interest formula: MRB = ((1 + mean_RPB) ^ 20) - 1
        mrb = (std::pow(1.0 + report.mean_rpb, blocks_per_month) - 1.0) * 100.0;
    }
    
    // Calculate MRP20B (Mean Return per 20TB) if we have enough data - for comparison
    double mrp20b = 0.0;
    if (report.successful_blocks() >= 20) {
        double twenty_tb_return = 1.0;
        for (int i = 0; i < 20 && i < static_cast<int>(report.block_results.size()); ++i) {
            twenty_tb_return *= (1.0 + report.block_results[i].return_per_block);
        }
        mrp20b = (twenty_tb_return - 1.0) * 100.0;
    }
    
    // ANSI color codes for enhanced visual formatting
    const std::string RESET = "\033[0m";
    const std::string BOLD = "\033[1m";
    const std::string DIM = "\033[2m";
    
    // Colors
    const std::string BLUE = "\033[34m";
    const std::string GREEN = "\033[32m";
    const std::string RED = "\033[31m";
    const std::string YELLOW = "\033[33m";
    const std::string CYAN = "\033[36m";
    const std::string MAGENTA = "\033[35m";
    const std::string WHITE = "\033[37m";
    
    // Background colors
    const std::string BG_BLUE = "\033[44m";
    const std::string BG_GREEN = "\033[42m";
    const std::string BG_RED = "\033[41m";
    const std::string BG_YELLOW = "\033[43m";
    const std::string BG_CYAN = "\033[46m";
    const std::string BG_DARK = "\033[100m";
    
    // Determine performance color based on Mean RPB
    std::string perf_color = RED;
    std::string perf_bg = "";
    if (report.mean_rpb > 0.001) {  // > 0.1%
        perf_color = GREEN;
        perf_bg = "";
    } else if (report.mean_rpb > -0.001) {  // -0.1% to 0.1%
        perf_color = YELLOW;
        perf_bg = "";
    }
    
    // Header with enhanced styling
    std::cout << "\n" << BOLD << BG_BLUE << WHITE << "╔══════════════════════════════════════════════════════════════════════════════════╗" << RESET << std::endl;
    std::cout << BOLD << BG_BLUE << WHITE << "║                        🎯 CANONICAL EVALUATION COMPLETE                          ║" << RESET << std::endl;
    std::cout << BOLD << BG_BLUE << WHITE << "╚══════════════════════════════════════════════════════════════════════════════════╝" << RESET << std::endl;
    
    // Run Information Section
    std::cout << "\n" << BOLD << CYAN << "📋 RUN INFORMATION" << RESET << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ " << BOLD << "Run ID:" << RESET << "       " << BLUE << run_id << RESET << std::endl;
    std::cout << "│ " << BOLD << "Strategy:" << RESET << "     " << MAGENTA << cfg.strategy_name << RESET << std::endl;
    std::cout << "│ " << BOLD << "Dataset:" << RESET << "      " << DIM << dataset_meta.file_path << RESET << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
    
    // Time Periods Section
    std::cout << "\n" << BOLD << CYAN << "📅 TIME PERIODS" << RESET << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    
    // Dataset Period
    if (dataset_meta.time_range_start > 0 && dataset_meta.time_range_end > 0) {
        double dataset_days = (dataset_meta.time_range_end - dataset_meta.time_range_start) / (1000.0 * 60.0 * 60.0 * 24.0);
        std::cout << "│ " << BOLD << "Dataset Period:" << RESET << " " << BLUE << to_iso_string(dataset_meta.time_range_start) 
                  << RESET << " → " << BLUE << to_iso_string(dataset_meta.time_range_end) << RESET << " " 
                  << DIM << "(" << std::fixed << std::setprecision(1) << dataset_days << " days)" << RESET << std::endl;
    }
    
    // Test Period (full available period)
    if (report.test_start_ts_ms > 0 && report.test_end_ts_ms > 0) {
        double test_days = (report.test_end_ts_ms - report.test_start_ts_ms) / (1000.0 * 60.0 * 60.0 * 24.0);
        std::cout << "│ " << BOLD << "Test Period:" << RESET << "    " << GREEN << to_iso_string(report.test_start_ts_ms) 
                  << RESET << " → " << GREEN << to_iso_string(report.test_end_ts_ms) << RESET << " " 
                  << DIM << "(" << std::fixed << std::setprecision(1) << test_days << " days)" << RESET << std::endl;
    }
    
    // TB Period (actual Trading Blocks period)
    if (report.successful_blocks() > 0 && !report.block_results.empty()) {
        uint64_t tb_start_ms = report.block_results[0].start_ts_ms;
        uint64_t tb_end_ms = report.block_results[report.successful_blocks() - 1].end_ts_ms;
        double tb_days = (tb_end_ms - tb_start_ms) / (1000.0 * 60.0 * 60.0 * 24.0);
        std::cout << "│ " << BOLD << "TB Period:" << RESET << "      " << YELLOW << to_iso_string(tb_start_ms) 
                  << RESET << " → " << YELLOW << to_iso_string(tb_end_ms) << RESET << " " 
                  << DIM << "(" << std::fixed << std::setprecision(1) << tb_days << " days, " << report.successful_blocks() << " TBs)" << RESET << std::endl;
    }
    
    std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
    
    // Trading Configuration Section
    std::cout << "\n" << BOLD << CYAN << "⚙️  TRADING CONFIGURATION" << RESET << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ " << BOLD << "Trading Blocks:" << RESET << "  " << YELLOW << report.successful_blocks() << RESET << "/" 
              << YELLOW << block_config.num_blocks << RESET << " TB " << DIM << "(480 bars each ≈ 8hrs)" << RESET << std::endl;
    std::cout << "│ " << BOLD << "Total Bars:" << RESET << "     " << WHITE << report.total_bars_processed << RESET << " " 
              << DIM << "(" << std::fixed << std::setprecision(1) << (report.total_bars_processed / 390.0) << " trading days)" << RESET << std::endl;
    std::cout << "│ " << BOLD << "Total Fills:" << RESET << "    " << CYAN << report.total_fills << RESET << std::endl;
    std::cout << "│ " << BOLD << "Trades per TB:" << RESET << "  " << CYAN << std::fixed << std::setprecision(1) << trades_per_tb << RESET << " " << DIM << "(≈Daily)" << RESET << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
    
    // **NEW**: Instrument Distribution with P&L Breakdown for Canonical Evaluation
    std::cout << "\n" << BOLD << CYAN << "🎯 INSTRUMENT DISTRIBUTION & P&L BREAKDOWN" << RESET << std::endl;
    std::cout << "┌────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ Instrument │  Total Volume  │  Net P&L       │  Fill Count    │ Avg Fill Size  │" << std::endl;
    std::cout << "├────────────┼────────────────┼────────────────┼────────────────┼────────────────┤" << std::endl;
    
    // Get instrument statistics from audit database
    std::map<std::string, double> instrument_volume;
    std::map<std::string, double> instrument_pnl;
    std::map<std::string, int> instrument_fills;
    
    // Query the audit database for fill events
    if (auto* db_recorder = dynamic_cast<audit::AuditDBRecorder*>(&audit)) {
        std::string run_id = db_recorder->get_run_id();
        sqlite3* db = db_recorder->get_db().get_db();
        
        std::string query = "SELECT symbol, qty, price, pnl_delta FROM audit_events WHERE run_id = ? AND kind = 'FILL' ORDER BY seq ASC";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
            
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                std::string symbol = (char*)sqlite3_column_text(stmt, 0);
                double qty = sqlite3_column_double(stmt, 1);
                double price = sqlite3_column_double(stmt, 2);
                double pnl_delta = sqlite3_column_double(stmt, 3);
                
                instrument_volume[symbol] += std::abs(qty * price);
                instrument_pnl[symbol] += pnl_delta;
                instrument_fills[symbol]++;
            }
            sqlite3_finalize(stmt);
        }
    }
    
    // **FIX P&L MISMATCH**: Get canonical total P&L from final equity
    double canonical_total_pnl = 0.0;
    double starting_capital = 100000.0; // Standard starting capital
    
    // Extract final equity from the last FILL event's note field (matches canonical evaluation)
    if (auto* db_recorder = dynamic_cast<audit::AuditDBRecorder*>(&audit)) {
        std::string run_id = db_recorder->get_run_id();
        sqlite3* db = db_recorder->get_db().get_db();
        
        std::string query = "SELECT note FROM audit_events WHERE run_id = ? AND kind = 'FILL' ORDER BY seq DESC LIMIT 1";
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(stmt, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                std::string note = (char*)sqlite3_column_text(stmt, 0);
                size_t eq_pos = note.find("eq_after=");
                if (eq_pos != std::string::npos) {
                    size_t start = eq_pos + 9; // Length of "eq_after="
                    size_t end = note.find(",", start);
                    if (end == std::string::npos) end = note.length();
                    std::string eq_str = note.substr(start, end - start);
                    try {
                        double final_equity = std::stod(eq_str);
                        canonical_total_pnl = final_equity - starting_capital;
                    } catch (...) {
                        // Fall back to sum of pnl_delta if parsing fails
                        canonical_total_pnl = 0.0;
                        for (const auto& [instrument, pnl] : instrument_pnl) {
                            canonical_total_pnl += pnl;
                        }
                    }
                }
            }
            sqlite3_finalize(stmt);
        }
    }
    
    // **FIX**: Display ALL expected instruments (including those with zero activity)
    double total_volume = 0.0;
    double total_instrument_pnl = 0.0; // Sum of individual instrument P&Ls
    int total_fills = 0;
    
    // **ENSURE ALL QQQ FAMILY INSTRUMENTS ARE SHOWN**
    std::vector<std::string> all_expected_instruments = {"PSQ", "QQQ", "TQQQ", "SQQQ"};
    
    for (const std::string& instrument : all_expected_instruments) {
        double volume = instrument_volume.count(instrument) ? instrument_volume[instrument] : 0.0;
        double pnl = instrument_pnl.count(instrument) ? instrument_pnl[instrument] : 0.0;
        int fills = instrument_fills.count(instrument) ? instrument_fills[instrument] : 0;
        double avg_fill_size = (fills > 0) ? volume / fills : 0.0;
        
        total_volume += volume;
        total_instrument_pnl += pnl;
        total_fills += fills;
        
        // Color coding
        const char* pnl_color = (pnl >= 0) ? GREEN.c_str() : RED.c_str();
        
        printf("│ %-10s │ %14.2f │ %s$%+13.2f%s │ %14d │ $%13.2f │\n",
               instrument.c_str(), volume,
               pnl_color, pnl, RESET.c_str(),
               fills, avg_fill_size);
    }
    
    std::cout << "├────────────┼────────────────┼────────────────┼────────────────┼────────────────┤" << std::endl;
    
    // Totals row - use canonical P&L for accuracy
    const char* canonical_pnl_color = (canonical_total_pnl >= 0) ? GREEN.c_str() : RED.c_str();
    printf("│ %-10s │ %14.2f │ %s$%+13.2f%s │ %14d │ $%13.2f │\n",
           "TOTAL", total_volume,
           canonical_pnl_color, canonical_total_pnl, RESET.c_str(),
           total_fills, (total_fills > 0) ? total_volume / total_fills : 0.0);
    
    // **IMPROVED P&L RECONCILIATION**: Show breakdown of realized vs unrealized P&L
    if (std::abs(total_instrument_pnl - canonical_total_pnl) > 1.0) {
        double unrealized_pnl = canonical_total_pnl - total_instrument_pnl;
        printf("│ %-10s │ %14s │ %s$%+13.2f%s │ %14s │ $%13s │\n",
               "Realized", "",
               (total_instrument_pnl >= 0) ? GREEN.c_str() : RED.c_str(), 
               total_instrument_pnl, RESET.c_str(), "", "");
        printf("│ %-10s │ %14s │ %s$%+13.2f%s │ %14s │ $%13s │\n",
               "Unrealized", "",
               (unrealized_pnl >= 0) ? GREEN.c_str() : RED.c_str(),
               unrealized_pnl, RESET.c_str(), "", "");
    }
    
    std::cout << "└────────────┴────────────────┴────────────────┴────────────────┴────────────────┘" << std::endl;
    
    // **NEW**: Transaction Cost Analysis to explain Mean RPB vs Net P&L relationship
    if (auto* db_recorder = dynamic_cast<audit::AuditDBRecorder*>(&audit)) {
        std::string run_id = db_recorder->get_run_id();
        sqlite3* db = db_recorder->get_db().get_db();
        
        // Calculate total transaction costs from FILL events
        double total_transaction_costs = 0.0;
        int sell_count = 0;
        
        std::string cost_query = "SELECT qty, price, note FROM audit_events WHERE run_id = ? AND kind = 'FILL' ORDER BY seq ASC";
        sqlite3_stmt* cost_stmt;
        if (sqlite3_prepare_v2(db, cost_query.c_str(), -1, &cost_stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_text(cost_stmt, 1, run_id.c_str(), -1, SQLITE_TRANSIENT);
            
            while (sqlite3_step(cost_stmt) == SQLITE_ROW) {
                double qty = sqlite3_column_double(cost_stmt, 0);
                double price = sqlite3_column_double(cost_stmt, 1);
                std::string note = (char*)sqlite3_column_text(cost_stmt, 2);
                
                // Extract fees from note (fees=X.XX format)
                size_t fees_pos = note.find("fees=");
                if (fees_pos != std::string::npos) {
                    size_t start = fees_pos + 5; // Length of "fees="
                    size_t end = note.find(",", start);
                    if (end == std::string::npos) end = note.find(")", start);
                    if (end == std::string::npos) end = note.length();
                    std::string fees_str = note.substr(start, end - start);
                    try {
                        double fees = std::stod(fees_str);
                        total_transaction_costs += fees;
                        if (qty < 0) sell_count++; // Count sell transactions (which have SEC/TAF fees)
                    } catch (...) {
                        // Skip if parsing fails
                    }
                }
            }
            sqlite3_finalize(cost_stmt);
        }
        
        // Display transaction cost breakdown
        std::cout << "\n" << BOLD << CYAN << "💰 TRANSACTION COST ANALYSIS" << RESET << std::endl;
        std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
        printf("│ Total Transaction Costs   │ %s$%11.2f%s │ SEC fees + FINRA TAF (sells only)    │\n", 
               RED.c_str(), total_transaction_costs, RESET.c_str());
        printf("│ Sell Transactions         │ %11d  │ Transactions subject to fees         │\n", sell_count);
        printf("│ Avg Cost per Sell         │ $%11.2f │ Average SEC + TAF cost per sell      │\n", 
               (sell_count > 0) ? total_transaction_costs / sell_count : 0.0);
        printf("│ Cost as %% of Net P&L      │ %10.2f%%  │ Transaction costs vs profit          │\n", 
               (canonical_total_pnl != 0) ? (total_transaction_costs / std::abs(canonical_total_pnl)) * 100.0 : 0.0);
        std::cout << "├─────────────────────────────────────────────────────────────────────────────────┤" << std::endl;
        std::cout << "│ " << BOLD << "Mean RPB includes all transaction costs" << RESET << "  │ Block-by-block returns are net       │" << std::endl;
        std::cout << "│ " << BOLD << "Net P&L is final equity difference" << RESET << "      │ Before/after capital comparison       │" << std::endl; 
        std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
    }
    
    // Performance insight
    if (canonical_total_pnl >= 0) {
        std::cout << GREEN << "✅ Net Positive P&L: Strategy generated profit across instruments" << RESET << std::endl;
    } else {
        std::cout << RED << "❌ Net Negative P&L: Strategy lost money across instruments" << RESET << std::endl;
    }
    
    // Performance Metrics Section - with color coding
    std::cout << "\n" << BOLD << CYAN << "📈 PERFORMANCE METRICS" << RESET << std::endl;
    std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ " << BOLD << "Mean RPB:" << RESET << "       " << perf_color << BOLD << std::fixed << std::setprecision(4) 
              << (report.mean_rpb * 100) << "%" << RESET << " " << DIM << "(Return Per Block - Net of Fees)" << RESET << std::endl;
    std::cout << "│ " << BOLD << "Std Dev RPB:" << RESET << "    " << WHITE << std::fixed << std::setprecision(4) 
              << (report.stdev_rpb * 100) << "%" << RESET << " " << DIM << "(Volatility)" << RESET << std::endl;
    std::cout << "│ " << BOLD << "MRB:" << RESET << "            " << perf_color << BOLD << std::fixed << std::setprecision(2) 
              << mrb << "%" << RESET << " " << DIM << "(Monthly Return)" << RESET << std::endl;
    std::cout << "│ " << BOLD << "ARB:" << RESET << "            " << perf_color << BOLD << std::fixed << std::setprecision(2) 
              << (report.annualized_return_on_block * 100) << "%" << RESET << " " << DIM << "(Annualized Return)" << RESET << std::endl;
    
    // Risk metrics
    std::string sharpe_color = (report.aggregate_sharpe > 1.0) ? GREEN : (report.aggregate_sharpe > 0) ? YELLOW : RED;
    std::cout << "│ " << BOLD << "Sharpe Ratio:" << RESET << "   " << sharpe_color << std::fixed << std::setprecision(2) 
              << report.aggregate_sharpe << RESET << " " << DIM << "(Risk-Adjusted Return)" << RESET << std::endl;
    
    std::string consistency_color = (report.consistency_score < 1.0) ? GREEN : (report.consistency_score < 2.0) ? YELLOW : RED;
    std::cout << "│ " << BOLD << "Consistency:" << RESET << "    " << consistency_color << std::fixed << std::setprecision(4) 
              << report.consistency_score << RESET << " " << DIM << "(Lower = More Consistent)" << RESET << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘" << std::endl;
    
    // Performance Summary Box
    std::cout << "\n" << BOLD;
    if (report.mean_rpb > 0.001) {
        std::cout << BG_GREEN << WHITE << "🎉 PROFITABLE STRATEGY ";
    } else if (report.mean_rpb > -0.001) {
        std::cout << BG_YELLOW << WHITE << "⚖️  NEUTRAL STRATEGY ";
    } else {
        std::cout << BG_RED << WHITE << "⚠️  LOSING STRATEGY ";
    }
    std::cout << RESET << std::endl;
    
    // End the main audit run
    std::int64_t end_ts = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    audit.event_run_end(end_ts, "{}");
    
    return report;
}

} // namespace sentio
```

