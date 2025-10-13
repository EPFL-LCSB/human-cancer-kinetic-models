#!/bin/sh
# Choose the correct Gurobi home path
export GHOME="/opt/gurobi912/linux64"

# Ensure the directory exists
mkdir -p ~/.gurobi

# Run the license activation
if [ -d "$GHOME" ]; then
    # $GHOME/bin/grbgetkey 8b82e418-0269-4963-bcfb-2cea2343ee9d && \
    echo "export GUROBI_HOME=${GHOME}" >> ~/.bashrc && \
    echo 'export PATH="${PATH}:${GUROBI_HOME}/bin"' >> ~/.bashrc && \
    echo 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"' >> ~/.bashrc
fi
