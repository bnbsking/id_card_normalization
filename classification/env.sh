#!/bin/bash

poetry run pip install -e . --no-deps

if [ "$MODE" = "dev" ]; then
  poetry run uvicorn serve:app --reload --host 0.0.0.0 --port 8072
else
  poetry run uvicorn serve:app --reload --host 0.0.0.0 --port 8062
fi

tail -f /dev/null
