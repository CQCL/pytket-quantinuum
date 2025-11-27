# Copy over poetry dependencies from theming repository
cp pytket-docs-theming/extensions/pyproject.toml .
cp pytket-docs-theming/extensions/poetry.lock .

# Install the docs dependencies. Creates a .venv directory in docs
poetry install

# NOTE: Editable wheel should be installed separately.
