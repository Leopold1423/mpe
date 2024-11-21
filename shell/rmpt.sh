#!/bin/bash

if [ -z "$1" ]; then
    echo "please offer a directory."
    exit 1
fi

if [ ! -d "$1" ]; then
    echo "the directory does not exist."
    exit 1
fi

find "$1" -type f -name "*.pt" -exec rm -f {} \;

echo "all *.pt in $1 are deleted"