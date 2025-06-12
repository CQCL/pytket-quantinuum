#!/bin/bash
rm -rf build/

# Move theming elements into the docs folder
cp -R pytket-docs-theming/_static .
cp -R pytket-docs-theming/quantinuum-sphinx .
cp pytket-docs-theming/conf.py .

# Get the name of the project
EXTENSION_NAME="$(basename "$(dirname `pwd`)")"

# Correct github link in navbar
sed -i '' 's#CQCL/tket#CQCL/'$EXTENSION_NAME'#' _static/nav-config.js

# Build the docs. Ensure we have the correct project title.
sphinx-build -W -b html -D html_title="$EXTENSION_NAME" . build || exit 1

sphinx-build -W -v -b coverage . build/coverage || exit 1

# Remove copied files. This ensures reusability.
rm -r _static 
rm -r quantinuum-sphinx
rm conf.py