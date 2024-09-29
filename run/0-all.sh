#!/bin/bash

for script in ./run/*.sh; do
    if [[ $script != "./run/0-all.sh" ]]; then
        echo "Running $script..."
        $script
    fi
done