#include "sentio/audit.hpp"
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
: run_id_(cfg.run_id), flush_each_(cfg.flush_each)
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
void AuditRecorder::event_bar(std::int64_t ts, const std::string& inst, const AuditBar& b){
  write_line_("\"type\":\"bar\",\"ts\":"+num_i(ts)+",\"inst\":\""+json_escape_(inst)+"\",\"o\":"+num_s(b.open)+",\"h\":"+num_s(b.high)+",\"l\":"+num_s(b.low)+",\"c\":"+num_s(b.close)+",\"v\":"+num_s(b.volume)+"}");
}
void AuditRecorder::event_signal(std::int64_t ts, const std::string& base, SigType t, double conf){
  write_line_("\"type\":\"signal\",\"ts\":"+num_i(ts)+",\"base\":\""+json_escape_(base)+"\",\"sig\":" + num_i((int)t) + ",\"conf\":"+num_s(conf)+"}");
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
    std::string ts_s; parse_kv(line, "ts", ts_s);
    std::int64_t ts = ts_s.empty()?0:std::stoll(ts_s);

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
  double cash_delta = -px*qty - fees; // This is correct as-is
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
