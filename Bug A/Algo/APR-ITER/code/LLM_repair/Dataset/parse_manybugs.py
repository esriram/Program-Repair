# Adopted from https://github.com/LASER-UMASS/AutomatedRepairApplicabilityData/blob/master/AnnotationScripts
# /ManyBugsSpecific/getPatchComplexity.py
import json
from difflib import unified_diff


def get_unified_diff(source, mutant):
    output = ""
    for line in unified_diff(source.split('\n'), mutant.split('\n'), lineterm=''):
        output += line + "\n"
    return output


def parse_manybugs(folder):
    with open(folder + "ManyBugs/single_function_repair.json") as f:
        data = json.load(f)
    return data


def clean_parse_manybugs(folder):
    r = parse_manybugs(folder)
    data = {}
    for key, value in r.items():
        data[key+".c"] = value
    return data


def clean_parse_manybugs_single_hunk(folder):
    r = parse_manybugs(folder)
    data = {}
    for key, value in r.items():
        diff_lines = get_unified_diff(value['buggy'], value['fix']).splitlines()
        remove, gap, add, single_hunk = False, False, False, True
        line_no = 0
        for line in diff_lines[2:]:
            if "@@" in line:
                rline = line.split(" ")[1][1:]
                line_no = int(rline.split(",")[0].split("-")[0])
            elif line.startswith("-"):
                if not remove:
                    start_line_no = line_no
                if gap:
                    single_hunk = False
                    break
                remove = True
                end_line_no = line_no
            elif line.startswith("+"):
                if not remove and not add:
                    start_line_no = line_no
                if not add:
                    end_line_no = line_no
                add = True
                if gap:
                    single_hunk = False
                    break
            else:
                if remove or add:
                    gap = True

            if not single_hunk:
                break
            line_no += 1

        if not single_hunk:
            pass
        else:
            data[key + ".c"] = value
            data[key + ".c"]['prefix'] = "\n".join(value['buggy'].splitlines()[:start_line_no - 2])
            data[key + ".c"]['suffix'] = "\n".join(value['buggy'].splitlines()[end_line_no - 2:])
    return data


def clean_parse_manybugs_single_line(folder):
    with open(folder + "ManyBugs/pass_tests.json", "r") as f:
        test = json.load(f)
    r = parse_manybugs(folder)
    data = {}
    for key, value in r.items():
        if not (key in test and "n1" in test[key]):
            continue
        diff_lines = get_unified_diff(value['buggy'], value['fix']).splitlines()
        remove, gap, add, single_hunk = False, False, False, True
        line_no = 0
        for line in diff_lines[2:]:
            if "@@" in line:
                rline = line.split(" ")[1][1:]
                line_no = int(rline.split(",")[0].split("-")[0])
            elif line.startswith("-"):
                if not remove:
                    start_line_no = line_no
                if gap:
                    single_hunk = False
                    break
                remove = True
                end_line_no = line_no
            elif line.startswith("+"):
                if not remove and not add:
                    start_line_no = line_no
                # if not add or (add and remove):
                end_line_no = line_no
                add = True
                if gap:
                    single_hunk = False
                    break
            else:
                if remove or add:
                    gap = True

            if not single_hunk:
                break
            line_no += 1

        if not single_hunk:
            # print("Not single hunk bug")
            pass
        else:
            if (end_line_no - start_line_no == 1 and remove) or end_line_no == start_line_no:
                data[key + ".c"] = value
                data[key + ".c"]['prefix'] = "\n".join(value['buggy'].splitlines()[:start_line_no - 2])
                data[key + ".c"]['suffix'] = "\n".join(value['buggy'].splitlines()[end_line_no - 2:])

            # print(data[key + ".c"]['suffix'])
    # print(len(data))
    return data
