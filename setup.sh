#!/bin/bash

# AutoML Harness Setup Script
# This script sets up the complete AutoML environment with proper Python environments

set -e  # Exit on any error

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
    
    # Check for available Python versions (prefer 3.11, fallback to 3.12/3.13)
    if command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
        log_success "Found Python 3.11 (recommended)"
    elif command -v python3.12 &> /dev/null; then
        PYTHON_CMD="python3.12"
        log_warning "Using Python 3.12 - some AutoML libraries may have compatibility issues"
        log_warning "For best compatibility, install Python 3.11: sudo pacman -S python311"
    elif command -v python3.13 &> /dev/null; then
        PYTHON_CMD="python3.13"
        log_warning "Using Python 3.13 - AutoML libraries may have significant compatibility issues"
        log_warning "Auto-sklearn may not work properly with Python 3.13"
        log_warning "For best compatibility, install Python 3.11: sudo pacman -S python311"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        log_warning "Using system Python 3 (version $PYTHON_VERSION)"
        log_warning "AutoML libraries work best with Python 3.11"
    else
        log_error "No Python 3 installation found. Please install Python 3.11 or later."
        exit 1
    fi
    
    log_success "System check passed - using $PYTHON_CMD"
}

# Install system dependencies
install_system_deps() {
    log_info "Checking system dependencies..."
    
    # Check if we can use package manager
    if command -v apt &> /dev/null; then
        log_info "Detected apt package manager"
        # We won't run sudo commands automatically, just inform user
        if ! dpkg -l | grep -q python3.11-dev; then
            log_warning "python3.11-dev not found. You may need to run:"
            log_warning "sudo apt update && sudo apt install -y python3.11-dev build-essential"
        fi
    elif command -v pacman &> /dev/null; then
        log_info "Detected pacman package manager"
        if ! pacman -Qi base-devel &> /dev/null; then
            log_warning "base-devel not found. You may need to run:"
            log_warning "sudo pacman -S base-devel"
        fi
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
}

# Create virtual environments
create_environments() {
    log_info "Creating Python virtual environments..."
    
    # Use the Python command determined in check_system
    
    # Create env-as (Auto-Sklearn environment)
    if [ ! -d "env-as" ]; then
        log_info "Creating env-as (Auto-Sklearn environment)..."
        $PYTHON_CMD -m venv env-as
        log_success "Created env-as environment"
    else
        log_info "env-as environment already exists"
    fi
    
    # Create env-tpa (TPOT + AutoGluon environment)  
    if [ ! -d "env-tpa" ]; then
        log_info "Creating env-tpa (TPOT + AutoGluon environment)..."
        $PYTHON_CMD -m venv env-tpa
        log_success "Created env-tpa environment"
    else
        log_info "env-tpa environment already exists"
    fi
}

# Install dependencies in env-as
install_env_as_deps() {
    log_info "Installing dependencies in env-as..."
    
    source env-as/bin/activate
    
    # Upgrade pip first
    pip install --upgrade pip
    
    # Install base scientific computing stack with version constraints for compatibility
    log_info "Installing base scientific packages..."
    if [[ "$PYTHON_CMD" == "python3.13" ]]; then
        log_warning "Installing compatible versions for Python 3.13..."
        pip install numpy scipy scikit-learn pandas matplotlib seaborn
        log_warning "Skipping Auto-sklearn installation due to Python 3.13 compatibility issues"
        log_warning "Auto-sklearn does not support Python 3.13 yet"
    else
        pip install numpy==1.24.3 scipy scikit-learn==1.3.2 pandas matplotlib seaborn
        
        # Install Auto-Sklearn
        log_info "Installing Auto-Sklearn..."
        pip install auto-sklearn==0.15.0
    fi
    
    # Install additional utilities
    pip install rich joblib
    
    deactivate
    log_success "env-as dependencies installed successfully"
}

# Install dependencies in env-tpa
install_env_tpa_deps() {
    log_info "Installing dependencies in env-tpa..."
    
    source env-tpa/bin/activate
    
    # Upgrade pip first
    pip install --upgrade pip
    
    # Install base scientific computing stack
    log_info "Installing base scientific packages..."
    pip install numpy scipy scikit-learn pandas matplotlib seaborn
    
    # Install TPOT (with setuptools for Python 3.13 compatibility)
    log_info "Installing TPOT..."
    if [[ "$PYTHON_CMD" == "python3.13" ]]; then
        pip install setuptools  # Required for pkg_resources
    fi
    pip install tpot
    
    # Install AutoGluon (with Python version check)
    if [[ "$PYTHON_CMD" == "python3.13" ]]; then
        log_warning "Skipping AutoGluon installation due to Python 3.13 compatibility issues"
        log_warning "AutoGluon has dependency conflicts with Python 3.13"
    else
        log_info "Installing AutoGluon..."
        pip install autogluon.tabular
    fi
    
    # Install additional utilities
    pip install rich joblib
    
    # Install XGBoost and LightGBM if not Python 3.13
    if [[ "$PYTHON_CMD" == "python3.13" ]]; then
        log_warning "Installing compatible ML libraries for Python 3.13..."
        # XGBoost should work with 3.13, but LightGBM might have issues
        pip install xgboost || log_warning "XGBoost installation failed"
        pip install lightgbm || log_warning "LightGBM installation failed"
    else
        pip install xgboost lightgbm
    fi
    
    deactivate
    log_success "env-tpa dependencies installed successfully"
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
    
    # Test env-as
    log_info "Testing env-as environment..."
    source env-as/bin/activate
    
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
import auto_sklearn.regression
import sklearn
import numpy as np
import pandas as pd
print('✓ Auto-Sklearn environment working correctly')
print(f'  - Auto-Sklearn version: {auto_sklearn.__version__}')
print(f'  - Scikit-learn version: {sklearn.__version__}')
print(f'  - NumPy version: {np.__version__}')
print(f'  - Pandas version: {pd.__version__}')
"
    fi
    deactivate
    
    # Test env-tpa
    log_info "Testing env-tpa environment..."
    source env-tpa/bin/activate
    
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
    deactivate
    
    log_success "All environments tested successfully"
}

# Create environment activation scripts
create_activation_scripts() {
    log_info "Creating environment activation scripts..."
    
    # Create activate-as.sh
    cat > activate-as.sh << 'EOF'
#!/bin/bash
# Activate Auto-Sklearn environment
echo "Activating Auto-Sklearn environment (env-as)..."
source env-as/bin/activate
echo "✓ Auto-Sklearn environment activated"
echo "Use 'deactivate' to exit the environment"
EOF
    chmod +x activate-as.sh
    
    # Create activate-tpa.sh  
    cat > activate-tpa.sh << 'EOF'
#!/bin/bash
# Activate TPOT + AutoGluon environment  
echo "Activating TPOT + AutoGluon environment (env-tpa)..."
source env-tpa/bin/activate
echo "✓ TPOT + AutoGluon environment activated"
echo "Use 'deactivate' to exit the environment"
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
    
    check_system
    install_system_deps
    # setup_pyenv  # Commented out to use system Python directly
    create_directories
    create_environments
    install_env_as_deps
    install_env_tpa_deps
    test_environments
    create_activation_scripts
    
    echo ""
    echo "=========================================="
    log_success "Setup completed successfully!"
    echo "=========================================="
    echo ""
    echo "Environment Usage:"
    echo "  • Auto-Sklearn:     ./activate-as.sh"
    echo "  • TPOT + AutoGluon: ./activate-tpa.sh"
    echo ""
    echo "Quick Start:"
    echo "  1. Run: ./activate-tpa.sh"
    echo "  2. Test: python orchestrator.py --all --time 300 --data DataSets/3/predictors_Hold\\ 1\\ Full_20250527_151252.csv --target DataSets/3/targets_Hold\\ 1\\ Full_20250527_151252.csv"
    echo ""
    echo "For more information, see README.md"
    echo ""
}

# Run main function
main "$@" 