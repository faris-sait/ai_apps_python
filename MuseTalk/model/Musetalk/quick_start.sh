#!/bin/bash

# MuseTalk Quick Start Script
# This script demonstrates basic usage of MuseTalk with RTX 3090

set -e

MUSETALK_ROOT=$(cd "$(dirname "$0")/../../" && pwd)
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

echo "=================================================="
echo "MuseTalk Quick Start - RTX 3090"
echo "=================================================="
echo ""
echo "MuseTalk Root: $MUSETALK_ROOT"
echo "Script Dir: $SCRIPT_DIR"
echo ""

# Function to print section headers
print_section() {
    echo ""
    echo "=================================================="
    echo "$1"
    echo "=================================================="
}

# Step 1: Check prerequisites
print_section "Step 1: Checking Prerequisites"

echo -n "Python 3: "
python3 --version

echo -n "CUDA: "
python3 -c "import torch; print(f'PyTorch {torch.__version__}')" || echo "Not found"

echo -n "GPU: "
python3 -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not found')" || echo "Not found"

# Step 2: Setup
print_section "Step 2: Running Setup"

read -p "Run setup to download models? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    bash "$SCRIPT_DIR/setup.sh"
else
    echo "Skipping setup. Make sure models are downloaded in ./models/"
fi

# Step 3: Prepare config
print_section "Step 3: Configuration"

echo "Config files location: $MUSETALK_ROOT/configs/inference/"
echo ""
echo "Available config files:"
ls -1 "$MUSETALK_ROOT/configs/inference/" 2>/dev/null || echo "No config files found"

echo ""
echo "Example config template: $SCRIPT_DIR/example_config.yaml"
echo ""

read -p "Use default config (test.yaml)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    CONFIG="$MUSETALK_ROOT/configs/inference/test.yaml"
else
    read -p "Enter config file path: " CONFIG
    if [ ! -f "$CONFIG" ]; then
        echo "Error: Config file not found: $CONFIG"
        exit 1
    fi
fi

echo "Using config: $CONFIG"

# Step 4: Version selection
print_section "Step 4: Select Model Version"

echo "v1.0: Original model (faster)"
echo "v1.5: Enhanced model (recommended, better quality)"
echo ""

read -p "Select version (1 for v1.0, 5 for v1.5, default: 5): " -n 1 VERSION
echo
case $VERSION in
    1) VERSION="v1.0" ;;
    5) VERSION="v1.5" ;;
    *) VERSION="v1.5" ;;
esac

echo "Selected version: $VERSION"

# Step 5: Mode selection
print_section "Step 5: Select Inference Mode"

echo "normal: Higher quality (slower)"
echo "realtime: Faster processing (real-time capable)"
echo ""

read -p "Select mode (n for normal, r for realtime, default: n): " -n 1 MODE
echo
case $MODE in
    n|N) MODE="normal" ;;
    r|R) MODE="realtime" ;;
    *) MODE="normal" ;;
esac

echo "Selected mode: $MODE"

# Step 6: Run inference
print_section "Step 6: Running Inference"

echo ""
echo "Starting video generation..."
echo "This may take several minutes depending on video length."
echo ""

cd "$MUSETALK_ROOT"
bash "$SCRIPT_DIR/generate_video.sh" "$VERSION" "$MODE" "$CONFIG"

EXIT_CODE=$?

# Step 7: Results
print_section "Step 7: Results"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Video generation completed successfully!"
    echo ""
    echo "Output location: $SCRIPT_DIR/output/"
    echo ""
    echo "Generated files:"
    ls -lh "$SCRIPT_DIR/output/" 2>/dev/null || echo "No output files found"
else
    echo "✗ Error during video generation. Check logs above."
    exit 1
fi

print_section "Next Steps"

echo ""
echo "1. Check output videos in: $SCRIPT_DIR/output/"
echo ""
echo "2. To use custom videos:"
echo "   - Copy your video to data/video/"
echo "   - Copy your audio to data/audio/"
echo "   - Create a config YAML file (see example_config.yaml)"
echo "   - Run: bash $SCRIPT_DIR/generate_video.sh v1.5 normal your_config.yaml"
echo ""
echo "3. To run with Python:"
echo "   python3 $SCRIPT_DIR/generate_video.py --version v1.5 --mode normal"
echo ""
echo "For more information, see: $SCRIPT_DIR/README.md"
echo ""
echo "=================================================="
