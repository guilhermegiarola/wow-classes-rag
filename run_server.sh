#!/bin/bash

# Navigate to the project root
cd "$(dirname "$0")"

# Run uvicorn from the project root, pointing to the api module
echo "Starting FastAPI server..."
python -m uvicorn api.controller:app --reload --host 0.0.0.0 --port 8000

# Alternative if the above doesn't work:
# cd api && uvicorn controller:app --reload --host 0.0.0.0 --port 8000

