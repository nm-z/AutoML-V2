# TODO

## Observed Errors

1. `python orchestrator.py --help` fails due to missing module `pandas`.
2. `make test` fails at `setup.sh` because it tries to activate `env-as`, which is not created.

## Action Items

- Update environment setup to ensure required Python packages (e.g., pandas) are installed before running the orchestrator.
- Modify `setup.sh` to either create `env-as` or skip the activation test if it is not needed.

