# A script that formats the outputs of the regression4j CLI

with open(f"projects_list.txt",'r') as f:
    project_names = f.readlines()

for i in range(114):
    with open(f"project_details/text{i}.txt",'r') as f:
        content = f.readlines()
        count = content[2].split(" ")[2]
        testcases = []
        for j in range(3,len(content)-3):
            line = content[j][2:].split(" ")
            if j == 3:
                testcases.append(line[11].strip())
            else:
                testcases.append(line[7].strip())

    with open(f"project_details/text{i}.txt",'w') as f:
        f.write(project_names[i])
        f.write(count+'\n')
        for testcase in testcases:
            f.write(testcase+'\n')