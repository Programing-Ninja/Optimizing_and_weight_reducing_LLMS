#!/bin/bash
# Launcher for sct_tq experiments.
# Ensures conda's libstdc++ (which has CXXABI_1.3.15) is on the loader path,
# preventing CXXABI errors that occur when the system libstdc++ is too old.
export LD_LIBRARY_PATH=/opt/miniconda3/lib:${LD_LIBRARY_PATH}
exec /scratch/DA24B039/sct_tq/.venv/bin/python "$@"
