import requests, zipfile
from io import BytesIO
import json
import os
import subprocess

data = json.load(open('Export.json','rb'))

count = 0

os.makedirs("data")

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

    os.makedirs(os.path.join("data",project_name),exist_ok=True)
    os.makedirs(os.path.join("data",project_name,str(i)),exist_ok=True)

    url_bic = f'https://github.com/{user_name}/{project_name}/archive/{bic}.zip'
    url_bfc = f'https://github.com/{user_name}/{project_name}/archive/{bfc}.zip'


    # Downloading the file by sending the request to the URL

    if os.path.exists(os.path.join("data",project_name,str(i),"fail")):
        print(f"{project_name} exists")
    else:
        print(f'Downloading {i}')
        req = requests.get(url_bic)
        req2 = requests.get(url_bfc)

        # extracting the zip file contents
        zipfile1 = zipfile.ZipFile(BytesIO(req.content))
        zipfile1.extractall(os.path.join("data",project_name,str(i)))

        zipfile2 = zipfile.ZipFile(BytesIO(req2.content))
        zipfile2.extractall(os.path.join("data",project_name,str(i)))

        print('Downloading Completed')

        curdir = os.path.join("data",project_name,str(i))
        curdirlist = os.listdir(os.path.join("data",project_name,str(i)))
        
        for folder in curdirlist:
            if folder == 'patch.txt':
                continue
            if folder.endswith(bic):
                os.rename(os.path.join(curdir,folder),os.path.join(curdir,"fail"))
            else:
                os.rename(os.path.join(curdir,folder),os.path.join(curdir,"success"))


