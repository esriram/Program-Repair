#!/usr/bin/env bash

# A script that uses regression4j's CLI to print information about the regressions of each project

i=0
while IFS= read -r line; do
    # printf '%s\n' "$line"    
    printf "use $line list\n^C" | ./CLi.sh > project_details/text$i.txt
    i=$((i+1))
done < projects_list.txt