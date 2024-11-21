#!/bin/bash

# usage:
# ./shell/kill.sh keyword

if [ $# -eq 0 ]; then
    PROCESS=`ps -ef | grep lishiwei | grep python | grep main.py | awk '{print $2}' | xargs kill -9`
else
    PROCESS=`ps -ef | grep lishiwei | grep python | grep main.py | grep $1 | awk '{print $2}' | xargs kill -9`
fi
