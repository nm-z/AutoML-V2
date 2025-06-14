# AGENT Instructions

## AutoML Engine Requirements
All AutoML orchestrations must always run using all three engine wrappers:
- `auto_sklearn_wrapper`
- `tpot_wrapper`
- `autogluon_wrapper`

These engines must be used together regardless of any other options or instructions.

## TODO Structure Requirements
**CRITICAL**: The TODO.md file uses a goal-oriented structure that must be preserved:

1. **Never revert the goal-oriented format** - Keep the "🎯 Primary Goals" structure with Goal 1 (Dataset 2) and Goal 2 (R² > 0.95)
2. **Never remove completed task markers** - Tasks marked with `[x]` should remain to track progress
3. **Add new tasks appropriately** - New tasks go under relevant goals or infrastructure sections
4. **Update completed tasks** - When completing work, mark tasks as `[x]` and add to "Major Milestones"

## Technical Requirements
1. **Preserve pyenv initialization** - All shell scripts must include the pyenv setup block from `run_all.sh`
2. **Maintain environment structure** - Use `automl-py310` and `automl-py311` environments as established
3. **Follow dataset focus** - Priority is Dataset 2 (D2) training and achieving R² > 0.95

## PR Guidelines
- Base all PRs on the current main branch
- Preserve existing TODO structure and completed task tracking
- Include relevant tests and documentation updates
- Align changes with Goal 1 or Goal 2 objectives
