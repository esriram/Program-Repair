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
        listOfLines.insert(lineNum+2,countSpaces(listOfLines[lineNum])*' '+"// END OF TRY STATEMENT\n")
        return

    # 2. Otherwise count curly brackets
    leftCurly = 0
    rightCurly = 0
    start = True
    while leftCurly != rightCurly or start:
        start = False
        if listOfLines[lineNum].strip().startswith("}") or listOfLines[lineNum].strip().endswith("}"):
            rightCurly+=1
        if listOfLines[lineNum].strip().startswith("{") or listOfLines[lineNum].strip().endswith("{"):
            leftCurly+=1
        lineNum+=1
        
    lineNum-=1
    lineList = list(listOfLines[lineNum])
    lineList.insert(listOfLines[lineNum].rfind("}")+1,"\n"+countSpaces(listOfLines[lineNum])*' '+"// END OF TRY STATEMENT\n"+countSpaces(listOfLines[lineNum])*' ')   
    listOfLines[lineNum] = "".join(lineList)

i=0
for line in listOfLines:
    if line.strip() == "// START OF TRY STATEMENT":
        countBrackets(i+1)
    i+=1

with open(sys.argv[1],"w") as file:
    for line in listOfLines:
        file.write(line)

