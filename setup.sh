#!/bin/bash

# AutoML Harness Setup Script
# This script sets up the complete AutoML environment with proper Python environments

set -eo pipefail

# Optional Auto-Sklearn environment
ENABLE_AS=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Linux
check_system() {
    log_info "Checking system compatibility..."
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        log_warning "This script is optimized for Linux. Continuing anyway..."
    fi
    
    # Prefer Python 3.11 but fall back to Python 3.10 if necessary
    if command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
        log_success "Found Python 3.11"
    elif command -v python3.10 &> /dev/null; then
        PYTHON_CMD="python3.10"
        log_warning "Python 3.11 not found; falling back to Python 3.10"
        ENABLE_AS=true
    else
        log_error "Python 3.11 or 3.10 is required but neither was found."
        log_info "For Ubuntu/Debian: sudo apt install python3.11 python3.11-venv python3.11-dev"
        log_info "For Arch: sudo pacman -S python311"
        exit 1
    fi
    
    PYTHON_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')
    if [ "$PYTHON_MINOR" -le 10 ]; then
        ENABLE_AS=true
    fi
    log_success "System check passed - using $PYTHON_CMD"
}

# Install system dependencies
install_system_deps() {
    log_info "Checking system dependencies..."
    
    # Detect OS and install appropriate packages
    if command -v apt &> /dev/null; then
        log_info "Detected Debian/Ubuntu system with apt"
        if ! dpkg -l | grep -q python3.11-dev; then
            log_info "Installing python3.11-dev and build-essential..."
            sudo apt update && sudo apt install -y python3.11-dev build-essential || {
                log_error "Failed to install system dependencies. Please run manually:"
                log_error "sudo apt update && sudo apt install -y python3.11-dev build-essential"
                exit 1
            }
        else
            log_success "Required packages already installed"
        fi
    elif command -v pacman &> /dev/null; then
        log_info "Detected Arch Linux system with pacman"
        if ! pacman -Qi base-devel &> /dev/null || ! pacman -Qi python &> /dev/null; then
            log_info "Installing base-devel and python..."
            sudo pacman -S --needed base-devel python || {
                log_error "Failed to install system dependencies. Please run manually:"
                log_error "sudo pacman -S --needed base-devel python"
                exit 1
            }
        else
            log_success "Required packages already installed"
        fi
    elif command -v dnf &> /dev/null; then
        log_info "Detected Fedora/RHEL system with dnf"
        if ! rpm -q python3.11-devel &> /dev/null || ! rpm -q gcc &> /dev/null; then
            log_info "Installing python3.11-devel and development tools..."
            sudo dnf install -y python3.11-devel gcc gcc-c++ make || {
                log_error "Failed to install system dependencies. Please run manually:"
                log_error "sudo dnf install -y python3.11-devel gcc gcc-c++ make"
                exit 1
            }
        else
            log_success "Required packages already installed"
        fi
    elif command -v yum &> /dev/null; then
        log_info "Detected CentOS/RHEL system with yum"
        if ! rpm -q python3.11-devel &> /dev/null || ! rpm -q gcc &> /dev/null; then
            log_info "Installing python3.11-devel and development tools..."
            sudo yum install -y python3.11-devel gcc gcc-c++ make || {
                log_error "Failed to install system dependencies. Please run manually:"
                log_error "sudo yum install -y python3.11-devel gcc gcc-c++ make"
                exit 1
            }
        else
            log_success "Required packages already installed"
        fi
    elif command -v zypper &> /dev/null; then
        log_info "Detected openSUSE system with zypper"
        if ! rpm -q python311-devel &> /dev/null || ! rpm -q gcc &> /dev/null; then
            log_info "Installing python311-devel and development tools..."
            sudo zypper install -y python311-devel gcc gcc-c++ make || {
                log_error "Failed to install system dependencies. Please run manually:"
                log_error "sudo zypper install -y python311-devel gcc gcc-c++ make"
                exit 1
            }
        else
            log_success "Required packages already installed"
        fi
    else
        log_warning "Unknown package manager. Please ensure you have:"
        log_warning "- Python 3.11 development headers"
        log_warning "- C/C++ compiler (gcc/clang)"
        log_warning "- Make and other build tools"
        log_warning "Press Enter to continue or Ctrl+C to abort..."
        read -r
    fi
}

# Setup pyenv if needed
setup_pyenv() {
    log_info "Checking pyenv installation..."
    
    if ! command -v pyenv &> /dev/null; then
        log_info "Installing pyenv..."
        curl https://pyenv.run | bash
        
        # Add pyenv to PATH for current session
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init -)"
        eval "$(pyenv virtualenv-init -)"
        
        log_warning "Please add the following to your ~/.bashrc or ~/.zshrc:"
        echo 'export PYENV_ROOT="$HOME/.pyenv"'
        echo 'export PATH="$PYENV_ROOT/bin:$PATH"'
        echo 'eval "$(pyenv init -)"'
        echo 'eval "$(pyenv virtualenv-init -)"'
    else
        log_success "pyenv is already installed"
    fi

    # --- NEW: guarantee pyenv-virtualenv is available ---
    if [ ! -d "$(pyenv root)/plugins/pyenv-virtualenv" ]; then
        log_info "Installing pyenv-virtualenv plugin..."
        git clone https://github.com/pyenv/pyenv-virtualenv.git \
            "$(pyenv root)"/plugins/pyenv-virtualenv
    fi

    # Ensure the plugin is loaded for the rest of the script
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"

    # Shell function provided by plugin
    alias venv_on='pyenv activate'
    alias venv_off='pyenv deactivate'
}

# Create pyenv virtual environments
create_environments() {
    log_info "Creating Python environments with pyenv..."

    if ! command -v pyenv &> /dev/null; then
        log_error "pyenv is required but not installed"
        exit 1
    fi

    # Always attempt to create automl-py311
    if ! pyenv versions --bare | grep -q "automl-py311"; then
        log_info "Creating automl-py311 (TPOT + AutoGluon)..."
        pyenv install -s 3.11 || log_warning "Python 3.11 installation failed or skipped, but continuing."
        pyenv virtualenv 3.11 automl-py311 || log_warning "automl-py311 virtualenv creation failed or skipped, but continuing."
    else
        log_warning "automl-py311 already exists"
    fi

    # Always attempt to create automl-py310
    if ! pyenv versions --bare | grep -q "automl-py310"; then
        log_info "Creating automl-py310 (Auto-Sklearn)..."
        pyenv install -s 3.10 || log_warning "Python 3.10 installation failed or skipped, but continuing."
        pyenv virtualenv 3.10 automl-py310 || log_warning "automl-py310 virtualenv creation failed or skipped, but continuing."
    else
        log_warning "automl-py310 already exists"
    fi
}

# Install dependencies in automl-py311
install_env_tpa_deps() {
    log_info "Installing dependencies in automl-py311..."

    if ! pyenv versions --bare | grep -q "automl-py311"; then
        log_warning "automl-py311 environment not found. Skipping TPOT + AutoGluon dependencies."
        return
    fi

    # Ensure we're in the correct environment
    pyenv deactivate || true # Deactivate if any environment is active
    pyenv activate automl-py311
    python -m pip install --upgrade pip

    offline_opts=()
    if [ -n "${OFFLINE_WHEELS_DIR:-}" ] && [ -d "$OFFLINE_WHEELS_DIR" ]; then
        offline_opts=(--no-index --find-links "$OFFLINE_WHEELS_DIR")
    elif [ -d wheels ]; then
        offline_opts=(--no-index --find-links wheels)
    fi

    python -m pip install "${offline_opts[@]}" --only-binary=:all: -r requirements-py311.txt \
        || { log_error "pip install failed"; venv_off; exit 1; }
}

# Install dependencies in automl-py310
install_env_as_deps() {
    if ! pyenv versions --bare | grep -q "automl-py310"; then
        log_warning "automl-py310 environment not found. Skipping Auto-Sklearn dependencies"
        return
    fi

    log_info "Installing dependencies in automl-py310..."

    # Ensure we're in the correct environment
    pyenv deactivate || true # Deactivate if any environment is active
    pyenv activate automl-py310
    python -m pip install --upgrade pip

    offline_opts=()
    if [ -n "${OFFLINE_WHEELS_DIR:-}" ] && [ -d "$OFFLINE_WHEELS_DIR" ]; then
        offline_opts=(--no-index --find-links "$OFFLINE_WHEELS_DIR")
    elif [ -d wheels ]; then
        offline_opts=(--no-index --find-links wheels)
    fi

    # Pull Autogluon + LightGBM + TPOT here (Python 3.10)
    sed -i '/autogluon.tabular/d' requirements-py310.txt
    sed -i '/xgboost/d' requirements-py310.txt
    sed -i '/lightgbm/d' requirements-py310.txt

    python -m pip install "${offline_opts[@]}" --only-binary=:all: -r requirements-py310.txt \
        || { log_error "pip install failed"; venv_off; exit 1; }
}

# Create necessary directories
create_directories() {
    log_info "Creating required directories..."
    
    # Ensure all required directories exist per the workspace rules
    mkdir -p components/models
    mkdir -p components/preprocessors/scalers
    mkdir -p components/preprocessors/dimensionality  
    mkdir -p components/preprocessors/outliers
    mkdir -p engines
    mkdir -p 05_outputs
    mkdir -p DataSets/{1,2,3}
    
    log_success "Directory structure created"
}

# Test environments
test_environments() {
    log_info "Testing environment installations..."

    # Test automl-py310 only if it exists
    if pyenv versions --bare | grep -q "automl-py310"; then
        log_info "Testing automl-py310 environment..."
        pyenv deactivate || true # Deactivate if any environment is active
        pyenv activate automl-py310

        if [[ "$PYTHON_CMD" == "python3.13" ]]; then
            python -c "
import sklearn
import numpy as np
import pandas as pd
print('✓ Base scientific environment working correctly (Auto-sklearn skipped for Python 3.13)')
print(f'  - Scikit-learn version: {sklearn.__version__}')
print(f'  - NumPy version: {np.__version__}')
print(f'  - Pandas version: {pd.__version__}')
"
        else
            python -c "
import autosklearn.regression
import sklearn
import numpy as np
import pandas as pd
print('✓ Auto-Sklearn environment working correctly')
print(f'  - Auto-Sklearn version: {autosklearn.__version__}')
print(f'  - Scikit-learn version: {sklearn.__version__}')
print(f'  - NumPy version: {np.__version__}')
print(f'  - Pandas version: {pd.__version__}')
"
        fi
        pyenv deactivate || true # Deactivate after testing
    else
        log_warning "automl-py310 environment not found. Skipping Auto-Sklearn test."
    fi

    # Test automl-py311
    log_info "Testing automl-py311 environment..."
    pyenv deactivate || true # Deactivate if any environment is active
    pyenv activate automl-py311
    
    if [[ "$PYTHON_CMD" == "python3.13" ]]; then
        python -c "
import tpot
import sklearn
import numpy as np
import pandas as pd
print('✓ TPOT environment working correctly (AutoGluon skipped for Python 3.13)')
print(f'  - TPOT version: {tpot.__version__}')
print(f'  - Scikit-learn version: {sklearn.__version__}')
print(f'  - NumPy version: {np.__version__}')
print(f'  - Pandas version: {pd.__version__}')
try:
    import xgboost as xgb
    print(f'  - XGBoost version: {xgb.__version__}')
except ImportError:
    print('  - XGBoost: Not available')
try:
    import lightgbm as lgb
    print(f'  - LightGBM version: {lgb.__version__}')
except ImportError:
    print('  - LightGBM: Not available')
"
    else
        python -c "
import tpot
import autogluon.tabular as ag
import sklearn
import numpy as np
import pandas as pd
print('✓ TPOT + AutoGluon environment working correctly')
print(f'  - TPOT version: {tpot.__version__}')
print(f'  - AutoGluon version: {ag.__version__}')
print(f'  - Scikit-learn version: {sklearn.__version__}')
print(f'  - NumPy version: {np.__version__}')
print(f'  - Pandas version: {pd.__version__}')
"
    fi
    pyenv deactivate || true # Deactivate after testing
    
    log_success "All environments tested successfully"
}

# Post-setup check for all installed libraries
post_setup_check() {
    log_info "Running post-setup checks to verify library installations..."

    ALL_LIBS_OK=true

    # Check automl-py311 libraries
    pyenv deactivate || true # Deactivate if any environment is active
    pyenv activate automl-py311
    REQUIRED_TPA_LIBS=(
        "numpy"
        "pandas"
        "scikit-learn"
        "joblib"
        "rich"
        "tpot"
        "autogluon.tabular"
        "xgboost"
        "lightgbm"
    )
    for lib in "${REQUIRED_TPA_LIBS[@]}"; do
        log_info "Checking $lib..."
        if ! python -c "import $lib" &> /dev/null; then
            log_error "✗ $lib is NOT installed or cannot be imported."
            ALL_LIBS_OK=false
        else
            log_success "✓ $lib is installed."
        fi
    done
    pyenv deactivate || true # Deactivate after checking

    # Check automl-py310 libraries if environment exists
    if pyenv versions --bare | grep -q "automl-py310"; then
        pyenv deactivate || true # Deactivate if any environment is active
        pyenv activate automl-py310
        REQUIRED_AS_LIBS=(
            "numpy"
            "pandas"
            "scikit-learn"
            "joblib"
            "rich"
            "autosklearn.regression"
        )
        for lib in "${REQUIRED_AS_LIBS[@]}"; do
            log_info "Checking $lib..."
            if ! python -c "import $lib" &> /dev/null; then
                log_error "✗ $lib is NOT installed or cannot be imported."
                ALL_LIBS_OK=false
            else
                log_success "✓ $lib is installed."
            fi
        done
        pyenv deactivate || true # Deactivate after checking
    else
        log_warning "automl-py310 environment not found. Skipping Auto-Sklearn library checks."
    fi

    if ! $ALL_LIBS_OK; then
        log_error "Post-setup check FAILED. Some required libraries are missing. Please review the errors above."
        exit 1
    else
        log_success "All required libraries verified successfully!"
    fi
}

# Create environment activation scripts
create_activation_scripts() {
    log_info "Creating environment activation scripts..."

    # Create activate-as.sh
    cat > activate-as.sh << 'EOF'
#!/bin/bash
# Activate Auto-Sklearn environment
echo "Activating Auto-Sklearn environment (automl-py310)..."
pyenv activate automl-py310
echo "✓ Auto-Sklearn environment activated"
echo "Use 'pyenv deactivate' to exit the environment"
EOF
    chmod +x activate-as.sh

    # Create activate-tpa.sh
    cat > activate-tpa.sh << 'EOF'
#!/bin/bash
# Activate TPOT + AutoGluon environment
echo "Activating TPOT + AutoGluon environment (automl-py311)..."
pyenv activate automl-py311
echo "✓ TPOT + AutoGluon environment activated"
echo "Use 'pyenv deactivate' to exit the environment"
EOF
    chmod +x activate-tpa.sh

    log_success "Activation scripts created"
}

# Main setup function
main() {
    echo "=========================================="
    echo "       AutoML Harness Setup Script       "
    echo "=========================================="
    echo ""

    for arg in "$@"; do
        if [ "$arg" = "--with-as" ]; then
            ENABLE_AS=true
        fi
    done
    
    check_system
    install_system_deps
    setup_pyenv
    create_directories
    create_environments
    install_env_tpa_deps
    if [ "$ENABLE_AS" = true ]; then
        install_env_as_deps
    fi
    test_environments
    post_setup_check
    create_activation_scripts
    
    echo ""
    echo "=========================================="
    log_success "Setup completed successfully!"
    echo "=========================================="
    echo ""
    echo "Environment Usage:"
    echo "  • TPOT + AutoGluon: ./activate-tpa.sh"
    echo "  • Auto-Sklearn:     ./activate-as.sh (optional)"
    echo ""
    echo "Quick Start:"
    echo "  1. Run: ./activate-tpa.sh"
    echo "  2. Test: python orchestrator.py --all --time 300 --data DataSets/3/predictors_Hold\\ 1\\ Full_20250527_151252.csv --target DataSets/3/targets_Hold\\ 1\\ Full_20250527_151252.csv"
    echo "  3. Optionally try Auto-Sklearn with ./activate-as.sh"
    echo ""
    echo "For more information, see README.md"
    echo ""
}

# Run main function
main "$@" 