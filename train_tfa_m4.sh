#!/bin/bash

# Apple M4 Optimized TFA Training Script
# Run this in a separate terminal for multi-regime TFA training

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="configs/tfa_m4_optimized.yaml"
LOG_FILE="training_m4_$(date +%Y%m%d_%H%M%S).log"

echo -e "${PURPLE}🍎 Apple M4 TFA Training Script${NC}"
echo -e "${PURPLE}================================${NC}"
echo -e "${CYAN}📁 Working Directory: ${SCRIPT_DIR}${NC}"
echo -e "${CYAN}⚙️  Configuration: ${CONFIG_FILE}${NC}"
echo -e "${CYAN}📝 Log File: ${LOG_FILE}${NC}"
echo ""

# Check if we're in the right directory
if [[ ! -f "train_tfa_multi_regime.py" ]]; then
    echo -e "${RED}❌ Error: train_tfa_multi_regime.py not found${NC}"
    echo -e "${YELLOW}💡 Make sure you're running this from the sentio_cpp directory${NC}"
    exit 1
fi

# Check if config exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${RED}❌ Error: Configuration file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Check Python environment
echo -e "${BLUE}🐍 Checking Python Environment...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Error: python3 not found${NC}"
    exit 1
fi

# Check PyTorch and MPS availability
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
if torch.backends.mps.is_available():
    print('✅ Apple Metal Performance Shaders ready for M4 acceleration')
else:
    print('⚠️  MPS not available - will use CPU (much slower)')
"

# Check Apple Silicon
echo -e "${BLUE}🍎 Checking Apple Silicon...${NC}"
MACHINE=$(uname -m)
CHIP_INFO=$(system_profiler SPHardwareDataType | grep "Chip" | head -1)
echo -e "${CYAN}   Machine: $MACHINE${NC}"
echo -e "${CYAN}   Chip: $CHIP_INFO${NC}"

if [[ "$MACHINE" == "arm64" ]]; then
    echo -e "${GREEN}✅ Apple Silicon detected - optimized for M4 performance${NC}"
else
    echo -e "${YELLOW}⚠️  Not Apple Silicon - performance may be different${NC}"
fi

# Check data files
echo -e "${BLUE}📊 Checking Training Data...${NC}"
MISSING_FILES=()

# Check historic data
if [[ ! -f "data/equities/QQQ_RTH_NH.csv" ]]; then
    MISSING_FILES+=("data/equities/QQQ_RTH_NH.csv")
fi

# Check future data tracks (explicit list to avoid bash version issues)
TRACK_FILES=(
    "data/future_qqq/future_qqq_track_01.csv"
    "data/future_qqq/future_qqq_track_02.csv"
    "data/future_qqq/future_qqq_track_03.csv"
    "data/future_qqq/future_qqq_track_04.csv"
    "data/future_qqq/future_qqq_track_05.csv"
    "data/future_qqq/future_qqq_track_06.csv"
    "data/future_qqq/future_qqq_track_07.csv"
    "data/future_qqq/future_qqq_track_08.csv"
    "data/future_qqq/future_qqq_track_09.csv"
)

for FILE in "${TRACK_FILES[@]}"; do
    if [[ ! -f "$FILE" ]]; then
        MISSING_FILES+=("$FILE")
    fi
done

if [[ ${#MISSING_FILES[@]} -gt 0 ]]; then
    echo -e "${RED}❌ Missing data files:${NC}"
    for file in "${MISSING_FILES[@]}"; do
        echo -e "${RED}   - $file${NC}"
    done
    echo -e "${YELLOW}💡 Make sure all historic and future QQQ data is available${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All training data files found${NC}"

# Show data summary
echo -e "${BLUE}📈 Data Summary:${NC}"
HISTORIC_LINES=$(wc -l < "data/equities/QQQ_RTH_NH.csv")
echo -e "${CYAN}   Historic data: $(($HISTORIC_LINES - 1)) bars (2022-2025)${NC}"

FUTURE_TOTAL=0
for FILE in "${TRACK_FILES[@]}"; do
    if [[ -f "$FILE" ]]; then
        LINES=$(wc -l < "$FILE")
        FUTURE_TOTAL=$((FUTURE_TOTAL + LINES - 1))
    fi
done
echo -e "${CYAN}   Future data: $FUTURE_TOTAL bars (9 MarS tracks)${NC}"
echo -e "${CYAN}   Total dataset: $(($HISTORIC_LINES + $FUTURE_TOTAL - 1)) bars${NC}"

# Memory check
echo -e "${BLUE}💾 System Memory:${NC}"
MEMORY_GB=$(system_profiler SPHardwareDataType | grep "Memory:" | awk '{print $2}')
echo -e "${CYAN}   Available: $MEMORY_GB${NC}"

if [[ "${MEMORY_GB%% *}" -ge 16 ]]; then
    echo -e "${GREEN}✅ Sufficient memory for M4 training${NC}"
else
    echo -e "${YELLOW}⚠️  Low memory - consider reducing batch size${NC}"
fi

# Confirm before starting
echo ""
echo -e "${PURPLE}🚀 Ready to Start M4-Optimized TFA Training${NC}"
echo -e "${PURPLE}===========================================${NC}"
echo -e "${YELLOW}This will train a Transformer model on ~1.16M bars across multiple market regimes${NC}"
echo -e "${YELLOW}Estimated time on Apple M4: 2-4 hours${NC}"
echo -e "${YELLOW}Hard time limit: 6 hours (configurable)${NC}"
echo ""
read -p "$(echo -e ${CYAN}Continue with training? [y/N]: ${NC})" -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Training cancelled${NC}"
    exit 0
fi

# Start training with logging
echo -e "${GREEN}🚀 Starting M4-Optimized TFA Training...${NC}"
echo -e "${CYAN}📝 Logging to: $LOG_FILE${NC}"
echo -e "${CYAN}⏰ Started at: $(date)${NC}"
echo ""

# Create a function to handle cleanup
cleanup() {
    echo -e "\n${YELLOW}🛑 Training interrupted${NC}"
    echo -e "${CYAN}📝 Check log file: $LOG_FILE${NC}"
    exit 1
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Run training with both console output and logging
python3 train_tfa_multi_regime.py --config "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"

# Check if training completed successfully
if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}🎉 M4 Training Completed Successfully!${NC}"
    echo -e "${GREEN}====================================${NC}"
    echo -e "${CYAN}⏰ Finished at: $(date)${NC}"
    echo -e "${CYAN}📝 Full log: $LOG_FILE${NC}"
    echo -e "${CYAN}📁 Model artifacts: artifacts/TFA/v2_m4_optimized/${NC}"
    echo ""
    echo -e "${PURPLE}🔍 Quick Results Summary:${NC}"
    
    # Extract key metrics from log
    if [[ -f "$LOG_FILE" ]]; then
        echo -e "${CYAN}📊 Training Summary:${NC}"
        grep -E "(Best validation loss|Total training time|Training completed)" "$LOG_FILE" | tail -3
        echo ""
        echo -e "${CYAN}📈 Final Model Info:${NC}"
        if [[ -f "artifacts/TFA/v2_m4_optimized/model.meta.json" ]]; then
            python3 -c "
import json
with open('artifacts/TFA/v2_m4_optimized/model.meta.json', 'r') as f:
    meta = json.load(f)
print(f\"Model Type: {meta.get('model_type', 'Unknown')}\")
print(f\"Features: {meta.get('feature_count', 'Unknown')}\")
print(f\"Sequence Length: {meta.get('sequence_length', 'Unknown')}\")
print(f\"Training Bars: {meta.get('training_bars', 'Unknown'):,}\")
print(f\"Best Val Loss: {meta.get('best_val_loss', 'Unknown'):.6f}\")
print(f\"Time Limit Reached: {meta.get('time_limit_reached', False)}\")
"
        fi
    fi
    
    echo ""
    echo -e "${PURPLE}💡 Next Steps:${NC}"
    echo -e "${CYAN}   1. Test the model with: strattest tfa --mode simulation${NC}"
    echo -e "${CYAN}   2. Check model artifacts in: artifacts/TFA/v2_m4_optimized/${NC}"
    echo -e "${CYAN}   3. Review training log: $LOG_FILE${NC}"
    
else
    echo ""
    echo -e "${RED}❌ Training Failed${NC}"
    echo -e "${RED}=================${NC}"
    echo -e "${CYAN}📝 Check log file for details: $LOG_FILE${NC}"
    echo -e "${YELLOW}💡 Common issues:${NC}"
    echo -e "${YELLOW}   - Insufficient memory (reduce batch_size in config)${NC}"
    echo -e "${YELLOW}   - Missing dependencies (check PyTorch MPS support)${NC}"
    echo -e "${YELLOW}   - Data file issues (verify all CSV files exist)${NC}"
    exit 1
fi
