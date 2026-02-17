#!/bin/bash

# Navigate to the script's directory
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Creating venv..."
    python3 -m venv venv
else
    echo "Venv already exists."
fi

source venv/bin/activate

echo "----------------------------------------------------------------------"
echo "Installing requirements from requirements.txt..."
echo "----------------------------------------------------------------------"
pip install -r requirements.txt

echo ""
echo "----------------------------------------------------------------------"
echo "Installing UI dependencies (npm install)..."
echo "----------------------------------------------------------------------"
cd training-ui
npm install
cd ..

echo ""
echo "----------------------------------------------------------------------"
echo "Installation Complete!"
echo "----------------------------------------------------------------------"
