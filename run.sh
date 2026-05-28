#!/usr/bin/env zsh

VERBOSE=1 CONFIG=jk5 uvicorn app.server:app --reload --port 8004 --host 0.0.0.0
