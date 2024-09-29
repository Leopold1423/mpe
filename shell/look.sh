#!/bin/bash

username="lishiwei"
count=$(pgrep -u $username -c python)
echo "number of python process of lishiwei: $count"

if [ $# -eq 0 ]; then
    ps -ef | grep lishiwei | grep python | grep main.py
else
    ps -ef | grep lishiwei | grep python | grep main.py | grep $1
fi