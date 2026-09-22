#!/usr/bin/env bash
# Back-compat wrapper. The demo lives in scripts/demo/.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/demo/run_demo.sh" "$@"
