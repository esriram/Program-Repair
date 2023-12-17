import sys

listOfLines = []
with open(sys.argv[1],'r') as file:
    for line in file.readlines():
        listOfLines.append(line)

def countSpaces(line):
    count = 0
    for s in line:
        if s == ' ':
            count+=1
        elif s != ' ':
            break
    return count
        

def countBrackets(lineNum):
    # 1. Check if this for loop is written without {}
    if not listOfLines[lineNum].strip().endswith("{") and not listOfLines[lineNum+1].strip().startswith("{"):
        listOfLines.insert(lineNum+2,countSpaces(listOfLines[lineNum])*' '+"// END OF IF STATEMENT\n")
        return

    # 2. Otherwise count curly brackets
    leftCurly = 0
    rightCurly = 0
    start = True
    foundElse = False
    while leftCurly != rightCurly or start:
        start = False
        if listOfLines[lineNum].strip().startswith("}") or listOfLines[lineNum].strip().endswith("}"):
            rightCurly+=1
            if leftCurly == rightCurly and (("}else{" in listOfLines[lineNum]) or ("} else {" in listOfLines[lineNum])):
            	foundElse = True
            	lineNum+=1
            	break
        if listOfLines[lineNum].strip().startswith("{") or listOfLines[lineNum].strip().endswith("{"):
            leftCurly+=1
        lineNum+=1
        
    lineNum-=1            
    lineList = list(listOfLines[lineNum])
    lineList.insert(listOfLines[lineNum].rfind("}")+1,"\n"+countSpaces(listOfLines[lineNum])*' '+"// END OF IF STATEMENT\n")   
    listOfLines[lineNum] = "".join(lineList)    
    
    if foundElse == True:
        listOfLines[lineNum] = listOfLines[lineNum].replace("else",countSpaces(listOfLines[lineNum])*' '+"// START OF ELSE STATEMENT\n"+countSpaces(listOfLines[lineNum])*' '+"else",1)

def countBracketsElseIf(lineNum):
    # 1. Check if this for loop is written without {}
    if not listOfLines[lineNum].strip().endswith("{") and not listOfLines[lineNum+1].strip().startswith("{"):
        if listOfLines[lineNum+2].strip() == "else":
            listOfLines.insert(lineNum+2,countSpaces(listOfLines[lineNum])*' '+"// START OF ELSE STATEMENT\n")
        listOfLines.insert(lineNum+2,countSpaces(listOfLines[lineNum])*' '+"// END OF ELSE-IF STATEMENT\n")
        
        return

    # 2. Otherwise count curly brackets
    leftCurly = 0
    rightCurly = 0
    start = True
    foundElse = False
    while leftCurly != rightCurly or start:
        start = False
        if listOfLines[lineNum].strip().startswith("}") or listOfLines[lineNum].strip().endswith("}"):
            rightCurly+=1
            if leftCurly == rightCurly and (("}else{" in listOfLines[lineNum]) or ("} else {" in listOfLines[lineNum])):
            	foundElse = True
            	lineNum+=1
            	break
        if listOfLines[lineNum].strip().startswith("{") or listOfLines[lineNum].strip().endswith("{"):
            leftCurly+=1
        lineNum+=1
        
    lineNum-=1            
    lineList = list(listOfLines[lineNum])
    lineList.insert(listOfLines[lineNum].find("}")+1,"\n"+countSpaces(listOfLines[lineNum])*' '+"// END OF ELSE-IF STATEMENT\n")   
    listOfLines[lineNum] = "".join(lineList)   
    
    if foundElse == True:
        listOfLines[lineNum] = listOfLines[lineNum].replace("else",countSpaces(listOfLines[lineNum])*' '+"// START OF ELSE STATEMENT\n"+countSpaces(listOfLines[lineNum])*' '+"else",1)

i=0
for line in listOfLines:
    if "else // START OF IF STATEMENT" in line:
        listOfLines[i] = listOfLines[i].replace("else // START OF IF STATEMENT","// START OF ELSE-IF STATEMENT",1)
        if listOfLines[i+1].strip().startswith("if"):
    	    listOfLines[i+1] = listOfLines[i+1].replace("if", "else if",1)
        countBracketsElseIf(i+1)
    elif "// START OF IF STATEMENT" in line:
        countBrackets(i+1)
    i+=1

with open(sys.argv[1],"w") as file:
    for line in listOfLines:
        file.write(line)

