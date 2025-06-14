# AGENT Instructions

## 🚨 CRITICAL CURRENT ISSUE (2025-06-13)
**AutoML Libraries Not Installed**: All three engines are currently failing due to missing libraries and falling back to LinearRegression with R² = -1.3353. The 4 critical tasks in TODO.md must be addressed before any other work.

## AutoML Engine Requirements
All AutoML orchestrations must always run using all three engine wrappers:
- `auto_sklearn_wrapper`
- `tpot_wrapper`
- `autogluon_wrapper`

These engines must be used together regardless of any other options or instructions.

**Current Status**: All engines failing due to missing libraries - see Critical Tasks 1-4 in TODO.md

## TODO Structure Requirements
**CRITICAL**: The TODO.md file uses a goal-oriented structure that must be preserved:

1. **Never revert the goal-oriented format** - Keep the "🎯 Primary Goals" structure with Goal 1 (Dataset 2) and Goal 2 (R² > 0.95)
2. **Never remove completed task markers** - Tasks marked with `[x]` should remain to track progress
3. **Add new tasks appropriately** - New tasks go under relevant goals or infrastructure sections
4. **Update completed tasks** - When completing work, mark tasks as `[x]` and add to "Major Milestones"
5. **Preserve critical task sections** - The "🚨 CRITICAL TASKS" section must remain at the top until resolved

## Technical Requirements
1. **Preserve pyenv initialization** - All shell scripts must include the pyenv setup block from `run_all.sh`
2. **Maintain environment structure** - Use `automl-py310` and `automl-py311` environments as established
3. **Follow dataset focus** - Priority is Dataset 2 (D2) training and achieving R² > 0.95
4. **Environment activation** - Always verify proper environment activation before running AutoML engines
5. **Library verification** - Use test_imports.py to verify library installations before training

## PR Guidelines
- Base all PRs on the current main branch
- Preserve existing TODO structure and completed task tracking
- Include relevant tests and documentation updates
- Align changes with Goal 1 or Goal 2 objectives
- **DO NOT revert activation script improvements** (PR #161 changes must be preserved)
- **DO NOT delete validation helpers** (validate_outputs.py must be preserved)
- **Focus on library installation issues** - Priority is fixing the critical AutoML library problems

## Recent PR Review Summary (2025-06-13)
- **Processed**: 8 PRs (#155-#163)
- **Merged**: 6 PRs (Makefile fix, D2 docs, env plugin, dataset matrix, activation scripts, validation helper)
- **Closed**: 3 PRs (merge conflicts or reverted changes)
- **Key Improvements**: Enhanced activation scripts, added validation tools, improved documentation
- **Critical Discovery**: AutoML libraries not installed, all engines using LinearRegression fallback

## Environment Troubleshooting
If AutoML engines are failing:
1. Run `python test_imports.py` to check library availability
2. Verify pyenv environment with `pyenv version`
3. Test activation scripts: `./activate-tpa.sh` and `./activate-as.sh`
4. Check if automl-py310 and automl-py311 environments exist: `pyenv versions`
5. Re-run setup if needed: `bash setup.sh`
