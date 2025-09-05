#pragma once
#include <string>
#include <filesystem>
#include <cstdlib>

namespace sentio {
enum class TickerFamily { Qqq, Bitcoin, Tesla };

inline const char** family_symbols(TickerFamily f, int& n) {
  static const char* QQQ[] = {"QQQ","TQQQ","SQQQ","PSQ"};
  static const char* BTC[] = {"BTCUSD","ETHUSD"};
  static const char* TSLA[]= {"TSLA","TSLQ"};
  switch (f) {
    case TickerFamily::Qqq: n=4; return QQQ;
    case TickerFamily::Bitcoin: n=2; return BTC;
    case TickerFamily::Tesla: n=2; return TSLA;
  }
  n=0; return nullptr;
}

inline std::string resolve_csv(const std::string& symbol,
                               const std::string& equities_root="data/equities",
                               const std::string& crypto_root="data/crypto") {
  namespace fs = std::filesystem;
  std::string up = symbol; for (auto& c: up) c = ::toupper(c);
  auto is_crypto = (up=="BTC"||up=="BTCUSD"||up=="ETH"||up=="ETHUSD");

  const char* env_root = std::getenv("SENTIO_DATA_ROOT");
  const char* env_suffix = std::getenv("SENTIO_DATA_SUFFIX");
  std::string base = env_root ? std::string(env_root) : (is_crypto ? crypto_root : equities_root);
  std::string suffix = env_suffix ? std::string(env_suffix) : std::string("");

  // Prefer suffixed file in base, then non-suffixed, then fallback to default roots
  std::string cand1 = base + "/" + up + suffix + ".csv";
  if (fs::exists(cand1)) return cand1;
  std::string cand2 = base + "/" + up + ".csv";
  if (fs::exists(cand2)) return cand2;
  std::string fallback_base = (is_crypto ? crypto_root : equities_root);
  std::string cand3 = fallback_base + "/" + up + suffix + ".csv";
  if (fs::exists(cand3)) return cand3;
  return fallback_base + "/" + up + ".csv";
}
} // namespace sentio

