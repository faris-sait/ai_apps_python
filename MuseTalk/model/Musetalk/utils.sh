#!/bin/bash

# MuseTalk Utility Functions
# Helper script for common operations

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MUSETALK_ROOT=$(cd "$SCRIPT_DIR/../../" && pwd)
OUTPUT_DIR="$SCRIPT_DIR/output"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ====================================
# Utility Functions
# ====================================

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# ====================================
# Check GPU Status
# ====================================

check_gpu() {
    print_header "GPU Status Check"

    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi not found. NVIDIA drivers may not be installed."
        return 1
    fi

    echo ""
    nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,compute_cap --format=csv,noheader
    echo ""

    python3 << 'EOF'
import torch
print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"Total Memory: {props.total_memory / 1e9:.2f} GB")
    print(f"Compute Capability: {props.major}.{props.minor}")
EOF

    echo ""
    return 0
}

# ====================================
# List Generated Videos
# ====================================

list_videos() {
    print_header "Generated Videos"

    if [ ! -d "$OUTPUT_DIR" ]; then
        print_error "Output directory not found: $OUTPUT_DIR"
        return 1
    fi

    if [ ! "$(ls -A $OUTPUT_DIR)" ]; then
        print_warning "No videos found in output directory"
        return 0
    fi

    echo ""
    echo "Location: $OUTPUT_DIR"
    echo ""

    local count=0
    local total_size=0

    echo "Videos:"
    for file in "$OUTPUT_DIR"/*.mp4 "$OUTPUT_DIR"/*.avi "$OUTPUT_DIR"/*.mov 2>/dev/null; do
        if [ -f "$file" ]; then
            local size=$(du -h "$file" | cut -f1)
            local duration=""

            if command -v ffprobe &> /dev/null; then
                duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1:novalue=1 "$file" 2>/dev/null)
                if [ ! -z "$duration" ]; then
                    duration=$(printf "%.2f" $duration)
                    duration="${duration}s"
                fi
            fi

            printf "  %-40s %6s %s\n" "$(basename "$file")" "$size" "$duration"

            local file_size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
            total_size=$((total_size + file_size))
            count=$((count + 1))
        fi
    done

    echo ""
    if [ $count -gt 0 ]; then
        local total_gb=$(echo "scale=2; $total_size / 1073741824" | bc)
        echo "Total: $count video(s), ~${total_gb} GB"
        print_success "Videos found and ready!"
    else
        print_warning "No video files found"
    fi

    echo ""
    return 0
}

# ====================================
# Clean Output Directory
# ====================================

clean_output() {
    print_header "Clean Output Directory"

    if [ ! -d "$OUTPUT_DIR" ]; then
        print_warning "Output directory doesn't exist: $OUTPUT_DIR"
        return 0
    fi

    local count=$(find "$OUTPUT_DIR" -type f | wc -l)

    if [ $count -eq 0 ]; then
        print_warning "Output directory is already empty"
        return 0
    fi

    echo ""
    echo "Found $count file(s) in output directory"
    echo "Directory: $OUTPUT_DIR"
    echo ""

    read -p "Are you sure you want to delete all files? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$OUTPUT_DIR"/*
        print_success "Output directory cleaned"
    else
        print_warning "Cleanup cancelled"
    fi

    echo ""
}

# ====================================
# System Information
# ====================================

system_info() {
    print_header "System Information"

    echo ""
    echo "Python:"
    python3 --version
    echo ""

    echo "PyTorch:"
    python3 -c "import torch; print(f'Version: {torch.__version__}')"
    echo ""

    echo "CUDA:"
    python3 -c "import torch; print(f'Available: {torch.cuda.is_available()}')"
    echo ""

    echo "Operating System:"
    uname -s
    echo ""

    echo "Processor:"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sysctl -n machdep.cpu.brand_string
    else
        lscpu | grep "Model name" | cut -d: -f2
    fi

    echo ""
}

# ====================================
# Help/Usage
# ====================================

show_help() {
    cat << EOF
MuseTalk Utility Functions

Usage: bash utils.sh [command]

Commands:
  check-gpu         Check GPU status and CUDA availability
  list-videos       List all generated videos in output directory
  clean-output      Remove all files from output directory
  system-info       Display system information
  help              Show this help message

Examples:
  bash utils.sh check-gpu
  bash utils.sh list-videos
  bash utils.sh clean-output

EOF
}

# ====================================
# Main Entry Point
# ====================================

main() {
    local command=${1:-"help"}

    case "$command" in
        check-gpu|gpu)
            check_gpu
            ;;
        list-videos|list)
            list_videos
            ;;
        clean-output|clean)
            clean_output
            ;;
        system-info|info)
            system_info
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
