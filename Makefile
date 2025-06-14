PHONY += setup test clean

setup:
	@echo "Setting up the AutoML Harness environment..."
	@bash setup.sh
	@echo "Setup complete. Remember to activate your environment: pyenv activate automl-py311"

test:
	@echo "Running tests (not yet implemented - will run post-setup checks)..."
	@bash setup.sh # Rerun setup.sh to trigger post_setup_check. Placeholder for actual tests
	# In a real scenario, you would have a `python -m pytest` or similar here

clean:
	@echo "Cleaning up generated files and environments..."
	@rm -rf env-as env-tpa 05_outputs
	@find . -name "__pycache__" -exec rm -rf {} + || true
	@find . -name ".pytest_cache" -exec rm -rf {} + || true
	@echo "Cleanup complete." 