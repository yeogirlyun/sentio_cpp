#include "sentio/transformer_model.hpp"
#include "sentio/feature_pipeline.hpp"
#include "sentio/transformer_strategy_core.hpp"
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <filesystem>

using namespace sentio;

struct EvalConfig {
    std::string data_path;
    std::string model_path;
    int sequence_length = 64;
    int feature_dim = 128;
    int max_samples = 100000; // cap for speed if needed
};

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " --data PATH --model PATH [--sequence-length N] [--feature-dim N] [--max-samples N]\n";
}

static bool load_bars_csv(const std::string& path, std::vector<Bar>& out) {
    if (!std::filesystem::exists(path)) {
        std::cerr << "Error: data file not found: " << path << std::endl;
        return false;
    }
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Error: cannot open data file: " << path << std::endl;
        return false;
    }
    std::string line; bool first=true;
    while (std::getline(f, line)) {
        if (first) { first=false; continue; }
        std::istringstream ss(line);
        std::string tok; std::vector<std::string> toks;
        while (std::getline(ss, tok, ',')) toks.push_back(tok);
        if (toks.size() < 5) continue;
        try {
            Bar b{};
            b.open = std::stof(toks[1]);
            b.high = std::stof(toks[2]);
            b.low = std::stof(toks[3]);
            b.close = std::stof(toks[4]);
            b.volume = toks.size() > 5 ? static_cast<uint64_t>(std::stoll(toks[5])) : 0;
            b.ts_utc = toks[0];
            b.ts_utc_epoch = 0;
            out.push_back(b);
        } catch (...) { continue; }
    }
    return !out.empty();
}

struct QualityMetrics {
    double mse = 0.0;
    double mae = 0.0;
    double accuracy = 0.0; // directional
    double correlation = 0.0;
};

static QualityMetrics evaluate_model_on_data(const EvalConfig& cfg) {
    // Model config
    TransformerConfig mcfg; mcfg.feature_dim = cfg.feature_dim; mcfg.sequence_length = cfg.sequence_length;
    TransformerModel model(mcfg);
    // Load model if exists
    if (std::filesystem::exists(cfg.model_path)) {
        try { model.load_model(cfg.model_path); } catch (const std::exception& e) {
            std::cerr << "Warning: failed to load model: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "Warning: model file not found, using random weights: " << cfg.model_path << std::endl;
    }
    model.eval();
    model.optimize_for_inference();

    // Feature pipeline
    TransformerConfig::Features fcfg; fcfg.normalization = TransformerConfig::Features::NormalizationMethod::Z_SCORE; fcfg.decay_factor = 0.999f;
    FeaturePipeline pipeline(fcfg);

    // Load data
    std::vector<Bar> bars; bars.reserve(100000);
    if (!load_bars_csv(cfg.data_path, bars)) {
        throw std::runtime_error("Failed to load data");
    }

    // Build samples
    std::vector<float> preds; preds.reserve(cfg.max_samples);
    std::vector<float> labels; labels.reserve(cfg.max_samples);

    int start = cfg.sequence_length;
    for (int i = start; i < static_cast<int>(bars.size()) - 1 && static_cast<int>(preds.size()) < cfg.max_samples; ++i) {
        // Build window [i-seq, i)
        std::vector<torch::Tensor> seq_feats;
        seq_feats.reserve(cfg.sequence_length);
        for (int j = i - cfg.sequence_length; j < i; ++j) {
            std::vector<Bar> single{bars[j]};
            auto feat = pipeline.generate_features(single).squeeze(0);
            if (feat.size(0) != cfg.feature_dim) { seq_feats.clear(); break; }
            seq_feats.push_back(feat);
        }
        if (static_cast<int>(seq_feats.size()) != cfg.sequence_length) continue;
        auto features = torch::stack(seq_feats, 0).unsqueeze(0);
        torch::NoGradGuard no_grad;
        auto out = model.forward(features).squeeze();
        float pred = out.item<float>();
        float c = static_cast<float>(bars[i].close);
        float n = static_cast<float>(bars[i+1].close);
        float label = (n - c) / std::max(1e-8f, c);
        preds.push_back(pred);
        labels.push_back(label);
    }

    QualityMetrics qm{};
    if (preds.empty()) return qm;

    // MSE, MAE, ACC, Corr
    double se=0, ae=0; int acc=0; double mp=0, ml=0;
    for (size_t k=0;k<preds.size();++k){
        double e = preds[k] - labels[k]; se += e*e; ae += std::abs(e);
        mp += preds[k]; ml += labels[k];
        bool sp = preds[k] >= 0.0f; bool sl = labels[k] >= 0.0f; if (sp==sl) acc++;
    }
    qm.mse = se / preds.size();
    qm.mae = ae / preds.size();
    qm.accuracy = static_cast<double>(acc) / preds.size();
    double mean_p = mp / preds.size(), mean_l = ml / preds.size();
    double num=0, dp=0, dl=0; 
    for (size_t k=0;k<preds.size();++k){ double ap=preds[k]-mean_p; double al=labels[k]-mean_l; num+=ap*al; dp+=ap*ap; dl+=al*al; }
    qm.correlation = (dp>0 && dl>0) ? (num / std::sqrt(dp*dl)) : 0.0;
    return qm;
}

static std::string grade_from_metrics(const QualityMetrics& q) {
    // Primary driver: accuracy and MSE; use correlation as tiebreaker
    struct Tier { double acc; double mse; const char* grade; };
    const std::vector<Tier> tiers = {
        {0.60, 1e-4,  "A+"},
        {0.58, 2e-4,  "A"},
        {0.56, 3e-4,  "A-"},
        {0.54, 5e-4,  "B+"},
        {0.52, 8e-4,  "B"}
    };
    for (const auto& t : tiers) {
        if (q.accuracy >= t.acc && q.mse <= t.mse) return t.grade;
    }
    return "not-usable";
}

int main(int argc, char* argv[]) {
    EvalConfig cfg;
    for (int i=1;i<argc;++i){
        std::string a = argv[i];
        if (a=="--help"){ print_usage(argv[0]); return 0; }
        if (a=="--data" && i+1<argc) { cfg.data_path = argv[++i]; continue; }
        if (a=="--model" && i+1<argc) { cfg.model_path = argv[++i]; continue; }
        if (a=="--sequence-length" && i+1<argc) { cfg.sequence_length = std::stoi(argv[++i]); continue; }
        if (a=="--feature-dim" && i+1<argc) { cfg.feature_dim = std::stoi(argv[++i]); continue; }
        if (a=="--max-samples" && i+1<argc) { cfg.max_samples = std::stoi(argv[++i]); continue; }
        std::cerr << "Unknown arg: " << a << std::endl; print_usage(argv[0]); return 1;
    }
    if (cfg.data_path.empty() || cfg.model_path.empty()) { print_usage(argv[0]); return 1; }

    try {
        auto qm = evaluate_model_on_data(cfg);
        auto grade = grade_from_metrics(qm);
        std::cout << "Training Quality Evaluation" << std::endl;
        std::cout << "  Data:   " << cfg.data_path << std::endl;
        std::cout << "  Model:  " << cfg.model_path << std::endl;
        std::cout << "  MSE:    " << qm.mse << std::endl;
        std::cout << "  MAE:    " << qm.mae << std::endl;
        std::cout << "  ACC%:   " << (qm.accuracy*100.0) << std::endl;
        std::cout << "  Corr:   " << qm.correlation << std::endl;
        std::cout << "  Grade:  " << grade << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl; return 1;
    }
}


