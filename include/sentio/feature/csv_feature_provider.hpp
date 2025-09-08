#pragma once
#include "sentio/feature/feature_provider.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace sentio {

struct CsvFeatureProvider : IFeatureProvider {
  std::string path_;
  std::vector<std::string> names_;
  int T_{64};

  explicit CsvFeatureProvider(std::string path, int T)
  : path_(std::move(path)), T_(T) {
    std::ifstream f(path_);
    if (!f) throw std::runtime_error("missing features csv: " + path_);
    std::string header;
    std::getline(f, header);
    std::stringstream hs(header);
    std::string col; int idx=0;
    while (std::getline(hs, col, ',')) {
      if (idx==0 && (col=="bar_index"||col=="idx")) { idx++; continue; }
      if (col=="ts" || col=="timestamp") { idx++; continue; }
      names_.push_back(col);
      idx++;
    }
  }

  FeatureMatrix get_features_for(const std::string& /*symbol*/) override {
    std::ifstream f(path_);
    if (!f) throw std::runtime_error("missing: " + path_);
    std::string line;
    std::getline(f, line); // header
    std::vector<float> buf; buf.reserve(1<<20);
    int64_t rows=0; const int64_t cols=(int64_t)names_.size();
    while (std::getline(f, line)) {
      if (line.empty()) continue;
      std::stringstream ss(line);
      std::string cell; int colidx=0; bool first=true; bool have_ts=false;
      // optional bar_index, optional timestamp, then features
      while (std::getline(ss, cell, ',')) {
        if (first) { first=false; continue; }
        if (!have_ts) { have_ts=true; continue; }
        buf.push_back(cell.empty()? 0.0f : std::stof(cell));
        colidx++;
      }
      if (colidx != cols) throw std::runtime_error("col mismatch in " + path_);
      rows++;
    }
    return FeatureMatrix{rows, cols, std::move(buf)};
  }

  std::vector<std::string> feature_names() const override { return names_; }
  int seq_len() const override { return T_; }
};

} // namespace sentio
