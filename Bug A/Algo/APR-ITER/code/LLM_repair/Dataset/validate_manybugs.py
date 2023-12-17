import bugzoo
import sys
import os
import json
import gc
import glob
from bugzoo import Patch
from difflib import unified_diff
sys.path.append(os.path.dirname(os.path.join(sys.path[0], '../'))) # Hack

from parse_manybugs import parse_manybugs

url = "http://127.0.0.1:8080"


def get_unified_diff(source, mutant, filename):
    output = ""
    for line in unified_diff(source.split('\n'), mutant.split('\n'), lineterm='', tofile=filename, fromfile=filename):
        output += line + "\n"
    return output


def check_passing_test():
    clean_result = {}
    with open("../ManyBugs/pass_tests.json", "r") as f:
        result = json.load(f)
    print(len(result))
    print(len([x in x for x in result.values() if len(x) > 0]))
    for bug, tests in result.items():
        if len(tests) == 0:
            continue
        if "n1" in tests:
            clean_result[bug] = tests
    print(len(clean_result))
    return clean_result


def validate_all_patches(folder, j_file):
    tests = check_passing_test()

    with open(folder + "/" + j_file, "r") as f:
        repair_dict = json.load(f)
    with open("../ManyBugs/single_function_repair.json", "r") as f:
        bug_dict = json.load(f)

    plausible = 0
    total = 0
    client = bugzoo.Client(url)

    for file in sorted(glob.glob(folder + "/*.c")):

        project = file.split("/")[-1].split("_")[0]
        bug_id = file.split("/")[-1].split("_")[1].split(".c")[0]
        bugname = project + "_" + bug_id
        if len(file.split("/")[-1].split("_")) == 2:
            index = 0
        else:
            index = int(file.split("/")[-1].split("_")[2].split(".c")[0])-1

        if bugname not in tests:
            continue

        if len(repair_dict[project + "_" + bug_id+".c"]) <= index:
            continue

        if repair_dict[project + "_" + bug_id+".c"][index]['finish_reason'] != "stop":
            continue
        if repair_dict[project + "_" + bug_id+".c"][index]['diff'] == "":
            continue

        print(bugname, index)

        with open(file, 'r') as f:
            patch = f.readlines()
        with open("../ManyBugs/buggy_programs/"+bugname+".c", "r") as f:
            source = f.readlines()

        start = bug_dict[bugname]['start']
        end = bug_dict[bugname]['end']
        patch = "".join(source[:start - 1] + patch + source[end:])
        source = "".join(source)
        diff = get_unified_diff(source, patch, bug_dict[bugname]['filename'])
        # print(diff)
        print(repair_dict[project + "_" + bug_id+".c"][index]['diff'])
        bug = client.bugs['manybugs:' + bugname.replace("_", ":")]
        container = client.containers.provision(bug)
        patch = Patch.from_unidiff(diff)
        client.containers.patch(container, patch)
        client.containers.compile(container)
        success = True
        for test in bug.tests:
            if str(test.name) not in tests[bugname] or "p" in test.name:
                continue
            r = client.containers.test(container, test)
            print(r.to_dict())
            if not r.to_dict()['passed']:
                success = False
                break
        if success:
            repair_dict[bugname+".c"][index]['valid'] = True
            plausible += 1

        del client.containers[container.uid]
        gc.collect()
        total += 1

    print(len(sorted(glob.glob(folder + "/*.c"))))
    print("{}/{} are plausible".format(plausible, total))

    with open(folder + "/" + j_file, "w") as f:
        json.dump(repair_dict, f)
