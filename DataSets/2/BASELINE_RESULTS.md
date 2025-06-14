# Dataset 2 Baseline Results

The orchestrator was executed with all three engine wrappers using a 60 second time limit on June 14, 2025. AutoGluon successfully completed training while TPOT did not finish and Auto-Sklearn was unavailable in this environment.

## Hold-out metrics
- R²: 0.8382
- RMSE: 0.00108
- MAE: 0.00064

AutoGluon achieved an average cross-validation R² of 0.949.

## Leaderboard After Optimization

| Step | R² |
|------|------|
| Baseline (AutoGluon) | 0.8382 |
| RandomForest Tuning | 0.9601 |
| Weighted Ensemble | **0.9601** |

Further details can be found in `D2_Optimization.md`.
