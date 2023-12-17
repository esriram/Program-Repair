import sys
import os
from multiprocessing import Pool


def process_file(filename):
    if not filename.endswith('.java'): return
    filename = f'spooned/{filename}'
    os.system(f"python3 pythonScripts/spoonChangeCommentFor.py {filename}")
    os.system(f"python3 pythonScripts/spoonChangeCommentForeach.py {filename}")
    os.system(f"python3 pythonScripts/spoonChangeCommentIf.py {filename}")
    os.system(f"python3 pythonScripts/spoonChangeCommentElse.py {filename}")
    os.system(f"python3 pythonScripts/spoonChangeCommentSwitch.py {filename}")
    os.system(f"python3 pythonScripts/spoonChangeCommentWhile.py {filename}")
    os.system(f"python3 pythonScripts/spoonChangeCommentTry.py {filename}")
    os.system(f"python3 pythonScripts/spoonChangeCommentCatch.py {filename}")

os.system("bash spoonProcessors.sh")
f_names = os.listdir('spooned')

pool = Pool(os.cpu_count())
for i, _ in enumerate(pool.imap_unordered(process_file, f_names), 1):
    sys.stderr.write('\rpercentage of files completed: {0:%}'.format(i/len(f_names)))
