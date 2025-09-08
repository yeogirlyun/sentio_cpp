#pragma once
#include "irule.hpp"
#include "sma_cross_rule.hpp"
#include "bbands_squeeze_rule.hpp"
#include "vwap_reversion_rule.hpp"
#include "opening_range_breakout_rule.hpp"
#include "momentum_volume_rule.hpp"
#include "ofi_proxy_rule.hpp"
#include <memory>
#include <string>

namespace sentio::rules {

inline std::unique_ptr<IRuleStrategy> make_rule(const std::string& name){
  if (name=="SMA_CROSS")             return std::make_unique<SMACrossRule>();
  if (name=="BBANDS_SQUEEZE_BRK")    return std::make_unique<BBandsSqueezeBreakoutRule>();
  if (name=="VWAP_REVERSION")        return std::make_unique<VWAPReversionRule>();
  if (name=="OPENING_RANGE_BRK")     return std::make_unique<OpeningRangeBreakoutRule>();
  if (name=="MOMENTUM_VOLUME")       return std::make_unique<MomentumVolumeRule>();
  if (name=="OFI_PROXY")             return std::make_unique<OFIProxyRule>();
  return {};
}

} // namespace sentio::rules


