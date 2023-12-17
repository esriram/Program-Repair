#!/bin/bash
java -classpath src/main/java/processors.jar:spoon-core-10.2.0-beta-21-jar-with-dependencies.jar spoon.Launcher -i ./test --processors spoon.processors.WhileProcessor:spoon.processors.ForProcessor:spoon.processors.IfProcessor:spoon.processors.SwitchProcessor:spoon.processors.ForEachProcessor:spoon.processors.CatchProcessor:spoon.processors.TryProcessor

for file in $(find spooned -name '*.java');
do
  python3 pythonScripts/spoonChangeCommentFor.py $file
  python3 pythonScripts/spoonChangeCommentForeach.py $file
  python3 pythonScripts/spoonChangeCommentIf.py $file
  python3 pythonScripts/spoonChangeCommentElse.py $file
  python3 pythonScripts/spoonChangeCommentSwitch.py $file
  python3 pythonScripts/spoonChangeCommentWhile.py $file
  python3 pythonScripts/spoonChangeCommentTry.py $file
  python3 pythonScripts/spoonChangeCommentCatch.py $file
done
