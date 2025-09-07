#pragma once
#include <cstdint>

namespace sentio {
struct Bar {
  std::int64_t ts_epoch_us{0};
  double open{0}, high{0}, low{0}, close{0}, volume{0};
};
}
