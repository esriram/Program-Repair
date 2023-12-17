import requests
import json
import os

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

    download_url = 'https://github.com/' + user_name + '/' + project_name + '/compare/' + bic + '..' + bfc + '.patch'

    os.makedirs(os.path.join("data",project_name),exist_ok=True)
    os.makedirs(os.path.join("data",project_name,str(i)),exist_ok=True)

    # print(download_url)
    response = requests.get(download_url)
    open(os.path.join("data",project_name,str(i),'patch')+'.txt','wb').write(response.content)