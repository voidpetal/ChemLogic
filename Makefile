.PHONY: update-deps test prepare distcheck clean develop update reinstall activate install-python clear-outputs bump

# Update dependencies and generate requirements.txt
update-deps: 
	pdm update && \
	pdm fix && \
	pdm export -o requirements.txt --without-hashes --pyproject --prod

# Run formatting, linting, and tests
test:
	ruff format && \
	ruff check . --fix && \
	pdm test

# Build the package
distcheck:
	pdm build

# Run tests, update dependencies, and perform distribution check
prepare: test update-deps distcheck clear-outputs

# Install Python 3.11 using pyenv if not already installed
install-python:
	@if ! pyenv versions | grep 3.11; then \
		echo "Python 3.11 not found. Installing..."; \
		pyenv install 3.11; \
	fi

# Create a PDM environment in the local folder with package requirements
develop: install-python
	pyenv local 3.11 && \
	pdm update && \
	pdm install

bump:
	pdm bump minor

clear-outputs:
	find . -name "*.ipynb" -exec ./.venv/bin/jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {} \;