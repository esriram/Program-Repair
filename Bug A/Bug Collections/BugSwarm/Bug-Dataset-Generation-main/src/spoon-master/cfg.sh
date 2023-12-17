#!/bin/bash
java -classpath src/main/java/processors.jar:spoon-core-10.2.0-beta-21-jar-with-dependencies.jar:src/main/java/controlflow.jar:src/main/java/jgrapht-core-0.9.2.jar spoon.Launcher -i ./test2 -p spoon.processors.ControlFlowProcessor
