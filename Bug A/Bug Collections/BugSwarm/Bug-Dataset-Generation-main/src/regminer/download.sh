#!/usr/bin/env bash

# A script that uses regression4j's CLI to automatically extract the buggy and working directories
# based on the information in project_details/ and (optionally) test them with maven test.

for j in {0..113} # loop through all 114 projects
do
    i=0
    count=0 # number of regressions in this project
    name=0 # name of the project
    newname=0 # formatted name of the project
    while IFS= read -r line; do
        # printf '%s\n' "$line"
        if [ $i == 0 ]
        then
            name=$line
            newname=$(echo $name | sed 's/\//_/' | tr '[:upper:]' '[:lower:]') # format the project name
            mkdir $1/data/$newname # create new directory to store all the bugs of this project
        elif [ $i == 1 ]
        then
            count=$line # gets the number of bugs in this project
        else
            if [ $i -gt $((count+1)) ]
            then
                break
            fi
            printf "use $name \ncheckout $((i-1))\n^C" | ./CLi.sh # checkout current regression with regminer CLI
            mkdir $1/data/$newname/$((i-1)) # creates new directory for this regression
            mv $1/transfer_cache/$newname/* $1/data/$newname/$((i-1)) # moves downloaded bugs to data/ directory

            # Maven test (optional), $line corresponds to the unit tests for this regression
            # cd $1/data/$newname/$((i-1))/ric
            # mvn test -Dtest=$line
            # cd $2
        fi
        i=$((i+1))
    done < project_details/text$j.txt
done