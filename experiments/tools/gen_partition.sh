#! /bin/bash

set -x
set -euo pipefail

PYTHON_CMD_PATH="python3"

${PYTHON_CMD_PATH} dist_pre.py --dataset ogbpa --data-root-path /workspace/work/ds4gnn/DATA --part-algo vcrandom    --n-parts 4 --num-hops 0
${PYTHON_CMD_PATH} dist_pre.py --dataset ogbpa --data-root-path /workspace/work/ds4gnn/DATA --part-algo vcoblivious --n-parts 4 --num-hops 0

${PYTHON_CMD_PATH} dist_pre.py --dataset ogbpr --data-root-path /workspace/work/ds4gnn/DATA --part-algo vcrandom --n-parts 4 --num-hops 0 --self-loop
${PYTHON_CMD_PATH} dist_pre.py --dataset ogbpr --data-root-path /workspace/work/ds4gnn/DATA --part-algo random --n-parts 4 --self-loop

