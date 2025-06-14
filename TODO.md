# AutoML Project TODO

## 🚨 IMPORTANT NOTICE FOR CONTRIBUTORS
**As of latest update**: All PRs must be based on this new goal-oriented structure. PRs #124-#127 were closed because they reverted this structure. Please ensure your PRs:
1. Start from the current main branch
2. Preserve this goal-oriented TODO format
3. Include actual dataset training results when relevant
4. Align with Goals 1 or 2 below

## 🎯 Primary Goals

### Goal 1: Successfully Train All Engines on Dataset 2 (D2)
**Objective**: Get AutoGluon, Auto-Sklearn, and TPOT all running successfully on `DataSets/2/D2-Predictors.csv` and `DataSets/2/D2-Targets.csv`

**Current Status**: 🔴 BLOCKED
- AutoGluon: ✅ Working (R²=0.8383 on holdout)
- Auto-Sklearn: ❌ Module not installed properly
- TPOT: ❌ Failing with unexpected config_dict argument

**Sub-tasks**:
- [ ] Fix Auto-Sklearn installation and environment issues
- [ ] Debug TPOT parameter validation errors
- [ ] Verify all three engines can complete training on D2
- [ ] Document training results and performance metrics

### Goal 2: Achieve R² > 0.95 on Dataset 2
**Objective**: Optimize hyperparameters and feature engineering to achieve high performance

**Current Status**: 🔴 NOT STARTED (blocked by Goal 1)
- Best current result: R²=0.8383 (AutoGluon)
- Target: R² > 0.95

**Sub-tasks**:
- [ ] Baseline all engines on D2 (requires Goal 1 completion)
- [ ] Implement hyperparameter tuning strategies
- [ ] Add feature engineering pipeline
- [ ] Experiment with ensemble methods
- [ ] Document optimization strategies and results

## 🔧 Infrastructure & Setup Tasks

### Environment Management
- [x] Setup script creates `automl-py310` and `automl-py311` pyenv environments automatically
- [x] Fixed `run_all.sh` pyenv initialization for non-interactive shells
- [ ] **ACTIVE**: Implement Python 3.10 graceful fallback when unavailable
- [ ] **ACTIVE**: Create offline wheel installation support for restricted networks

### Testing & Validation
- [x] Added `--tree` flag to orchestrator for artifact directory display
- [x] Smoke test passes for basic orchestrator functionality
- [ ] **ACTIVE**: Improve smoke test documentation and error handling
- [ ] Verify `run_all.sh` works with all dataset combinations
- [ ] Add integration tests for all three engines

### Code Quality & Maintenance
- [x] Resolved scikit-learn version conflicts between engines
- [x] Fixed orchestrator duration calculation AttributeError
- [ ] **ACTIVE**: Fix Makefile indentation issues for `make test`
- [ ] **ACTIVE**: Enhance console logging with rich.tree progress display
- [ ] **ACTIVE**: Add TPOT parameter validation improvements

## 📊 Dataset Training Tasks

### Dataset 1 (D1) - Reference Implementation
- [x] All engines successfully train on D1
- [x] Baseline performance metrics established
- [x] Used as smoke test dataset in `run_all.sh`

### Dataset 2 (D2) - Primary Focus
- [x] **HIGH PRIORITY**: Create reproducible training script for D2 (`run_d2.sh`)
- [ ] Complete successful training run with all engines
- [ ] Document training results and failure modes
- [ ] Compare performance across engines
- [ ] Identify optimization opportunities
- [ ] Provide offline dependency install instructions to unblock Dataset 2 training

### Dataset 3 (D3) - Future Work
- [ ] Initial training runs with all engines
- [ ] Performance baseline establishment
- [ ] Comparison with D1 and D2 results

## 📋 Completed Tasks Archive

### Major Milestones
- [x] Git LFS setup for large files (`.pkl`, `.json`, `DataSets/`, `05_outputs/`)
- [x] Added run_all.sh for 60-second smoke testing
- [x] Systematic PR review and cleanup (34 PRs total: #94-#129, merged #129)
- [x] Added offline setup documentation for restricted environments
- [x] Applied `deactivate` to `pyenv deactivate` fix in setup.sh
- [x] Restructured TODO with goal-oriented delegation system
- [x] Created Dataset 2 training script (`run_d2.sh`) with proper pyenv initialization

### Bug Fixes & Improvements
- [x] Fixed orchestrator AttributeError for duration calculation
- [x] Resolved scikit-learn version conflicts (>=1.4.2,<1.6)
- [x] Added pyenv initialization to run_all.sh for non-interactive shells
- [x] Enhanced artifact directory tree display with --tree flag

## 🚨 Current PR Status

**Active PRs**: 0 open (all current PRs closed due to TODO structure conflicts)
- PRs #124-#127 closed: reverted new goal-oriented TODO structure
- Team should create new PRs based on current main branch and goal structure
- Focus on Goals 1 & 2 with actual dataset training results

## 📝 Notes for Contributors

1. **Dataset Training Focus**: PRs should include actual training results and performance metrics
2. **Goal Alignment**: All work should contribute to Goals 1 or 2 above
3. **Branch Management**: Always start from latest main branch
4. **Documentation**: Include training results, error logs, and performance data in PRs
5. **Testing**: Verify changes don't break existing pyenv initialization or smoke tests
6. **TODO Format**: Preserve this goal-oriented structure in any TODO modifications

