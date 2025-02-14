#!/bin/bash
while true
do
    rocm-smi -u --showmemuse --json
    sleep 1
done
