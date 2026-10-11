#!/bin/bash

poetry run pip install -e . --no-deps

if [ "$MODE" = "dev" ]; then
  poetry run uvicorn serve:app --reload --host 0.0.0.0 --port 8073
else
  poetry run uvicorn serve:app --reload --host 0.0.0.0 --port 8063
fi

tail -f /dev/null
