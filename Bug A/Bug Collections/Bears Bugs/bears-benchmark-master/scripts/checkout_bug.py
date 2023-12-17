import os
import sys
import subprocess
import json
import argparse

from config import *

parser = argparse.ArgumentParser(description='Script to check out one bug from Bears')
parser.add_argument('--bugId', help='The ID of the bug to be checked out', required=True, metavar='')
parser.add_argument('--workspace', help='The path to a folder to store the checked out bug', required=False, metavar='')
args = parser.parse_args()

BUG_ID = args.bugId
WORKSPACE = args.workspace

if WORKSPACE == None: WORKSPACE = "./workspace"
cmd = "mkdir -p %s" % WORKSPACE

subprocess.call(cmd, shell=True)

BUG_FOLDER_PATH = os.path.join(WORKSPACE, BUG_ID)
if os.path.isdir(BUG_FOLDER_PATH):
    print("The bug %s has already been checked out." % BUG_ID)
    sys.exit()

print("Checking out the bug %s..." % BUG_ID)

bugs = None
if os.path.exists(os.path.join(BEARS_PATH, BEARS_BUGS)):
    with open(os.path.join(BEARS_PATH, BEARS_BUGS), 'r') as f:
        try:
            bugs = json.load(f)
        except Exception as e:
            print("got %s on json.load()" % e)
            sys.exit()

BUG_BRANCH_NAME = None
if bugs is not None:
    for bug in bugs:
        if bug['bugId'].lower() == BUG_ID.lower():
            BUG_BRANCH_NAME = bug['bugBranch']
            break

if BUG_BRANCH_NAME is None:
    print("There is no bug with the ID %s" % BUG_ID)
    sys.exit()
    
print("Checking out the branch %s..." % BUG_BRANCH_NAME)

# create a folder for the bug in the workspace
cmd = "mkdir %s" % BUG_FOLDER_PATH
subprocess.call(cmd, shell=True)

# check out the branch containing the bug
cmd = "cd %s; git reset .; git checkout -- .; git clean -f; git checkout %s;" % (BEARS_PATH, BUG_BRANCH_NAME)
subprocess.call(cmd, shell=True)

# check out buggy commit from the branch containing the bug
cmd = "cd %s; git log --format=format:%%H --grep='Changes in the tests';" % BEARS_PATH
BUGGY_COMMIT = subprocess.check_output(cmd, shell=True).decode("utf-8")
cmd = "cd %s; git branch;" % BEARS_PATH
print("BUGGY_COMMIT",BUGGY_COMMIT)

cmd = "cd %s; git checkout %s;" % (BEARS_PATH, BUGGY_COMMIT)
subprocess.call(cmd, shell=True)

# copy all files to the bug folder
cmd = "find . -mindepth 1 -maxdepth 1 -not -name workspace -print0 | xargs -0 -I{} cp -r {} %s;" % ( BUG_FOLDER_PATH)
subprocess.call(cmd, shell=True)

# check out master
cmd = "cd %s; git reset .; git checkout -- .; git clean -f; git checkout master;" % BEARS_PATH
subprocess.call(cmd, shell=True)

print("The bug %s was checked out." % BUG_ID)
