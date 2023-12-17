#!/bin/bash
for file in $(find spooned -name '*.java');
do
  java -jar google-java-format-1.15.0-all-deps.jar --replace $file
done
