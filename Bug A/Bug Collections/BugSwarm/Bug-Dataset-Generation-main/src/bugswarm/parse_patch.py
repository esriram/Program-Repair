import shutil
import os
import json

data = json.load(open('Export.json','rb'))

for i in range(len(data)):
    if i > 6:
        break

    if data[i]['lang'] != 'Java':
        continue

    url = data[i]['diff_url'][8:]
    components = url.split("/")
    user_name = components[1]
    project_name = components[2]
    commits = components[4].split("..")
    bic = commits[0]
    bfc = commits[1]

    os.makedirs(os.path.join("data",project_name,str(i),'target'),exist_ok=True)
    os.makedirs(os.path.join("data",project_name,str(i),'target','fail'),exist_ok=True)
    os.makedirs(os.path.join("data",project_name,str(i),'target','success'),exist_ok=True)

    if os.path.exists(os.path.join("data",project_name,str(i),'patch.txt')):
        with open(os.path.join("data",project_name,str(i),'patch.txt')) as f:
            content = f.readlines()
            for line in content:
                if line.startswith("diff --git"):
                    # print(line)
                    tokens = line.split(" ")

                    bic_path = tokens[2] # a/.../...: buggy
                    bic_path = 'fail'+bic_path[1:].replace("\n",'')
                    bic_path_contents = bic_path.split("/")
                    bic_path_final = ''
                    for ppp in bic_path_contents:
                        bic_path_final = os.path.join(bic_path_final,ppp)

                    bfc_path = tokens[3] # b/.../...: success
                    bfc_path = 'success'+bfc_path[1:].replace("\n",'')
                    bfc_path_contents = bfc_path.split("/")
                    bfc_path_final = ''
                    for ppp in bfc_path_contents:
                        bfc_path_final = os.path.join(bfc_path_final,ppp)

                    if os.path.exists(os.path.join("data",project_name,str(i),bic_path_final)) and os.path.exists(os.path.join("data",project_name,str(i),bfc_path_final)):
                        shutil.copy(os.path.join("data",project_name,str(i),bic_path_final),os.path.join("data",project_name,str(i),'target','fail'))
                        shutil.copy(os.path.join("data",project_name,str(i),bfc_path_final),os.path.join("data",project_name,str(i),'target','success'))
    f.close()



                    



    