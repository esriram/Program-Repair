#!/bin/bash

for file in $(find test -name '*.java');
do
  echo $file
  java -classpath src/main/java/processors.jar:spoon-core-10.2.0-beta-21-jar-with-dependencies.jar spoon.Launcher -i $file --processors spoon.processors.WhileProcessor:spoon.processors.ForProcessor:spoon.processors.IfProcessor:spoon.processors.SwitchProcessor:spoon.processors.ForEachProcessor:spoon.processors.CatchProcessor:spoon.processors.TryProcessor
done
