import json
from difflib import unified_diff


def get_unified_diff(source, mutant):
    output = ""
    for line in unified_diff(source.split("\n"), mutant.split("\n"), lineterm=""):
        output += line + "\n"
    return output


def check_d4j_2(bug, d4j_2=False):
    is_d4j_2 = True
    if (
        "Time" in bug
        or "Math" in bug
        or "Mockito" in bug
        or "Chart" in bug
        or "Lang" in bug
    ):
        is_d4j_2 = False
    elif "Closure" in bug:
        if int(bug.split(".java")[0].split("-")[-1]) <= 133:
            is_d4j_2 = False

    return is_d4j_2 == d4j_2


def parse_d4j_12_full_with_repo_rare_token(project_identifier: str):
    with open(
        "Dataset/defects4j_1.2_refined_with_special_tokens_current_file_Levenshtein_ratio.json",
        "r",
    ) as f:
        result = json.load(f)

    cleaned_result = {}
    for k, v_s in result.items():
        if project_identifier is not None and project_identifier not in k:
            continue
        for v in v_s:
            lines = v["context"]["buggy"].splitlines()
            # leading_white_space = len(lines[0]) - len(lines[0].lstrip())
            leading_white_space = 0
            cleaned_result_v = {
                "buggy": "\n".join([line[leading_white_space:] for line in lines])
            }
            lines = v["context"]["prefix"].splitlines()
            cleaned_result_v["prefix"] = "\n".join(
                [line[leading_white_space:] for line in lines]
            )
            lines = v["context"]["suffix"].splitlines()
            cleaned_result_v["suffix"] = "\n".join(
                [line[leading_white_space:] for line in lines]
            )
            lines = v["context"]["fix"].splitlines()
            # leading_white_space = len(lines[0]) - len(lines[0].lstrip())
            leading_white_space = 0
            cleaned_result_v["fix"] = "\n".join(
                [line[leading_white_space:] for line in lines]
            )
            buggy_line = remove_suffix(
                remove_prefix(cleaned_result_v["buggy"], cleaned_result_v["prefix"]),
                cleaned_result_v["suffix"],
            ).replace("\n", "")
            cleaned_result_v["buggy_line"] = buggy_line
            cleaned_result_v["tokens"] = v["tokens"]
            cleaned_result_v["place_num"] = len(v_s)
            cleaned_result[
                k + "-place-" + str(v_s.index(v)) + ".java"
            ] = cleaned_result_v

    result = {k: v for k, v in cleaned_result.items() if check_d4j_2(k, False)}

    return result


def parse_d4j_2_full_with_repo_rare_token(project_identifier: str):
    with open(
        "Dataset/defects4j_2.0_single_line_with_special_tokens_current_file_Levenshtein_ratio.json",
        "r",
    ) as f:
        result = json.load(f)

    cleaned_result = {}
    for k, v_s in result.items():
        if project_identifier is not None and project_identifier not in k:
            continue
        for v in v_s:
            lines = v["context"]["buggy"].splitlines()
            # leading_white_space = len(lines[0]) - len(lines[0].lstrip())
            leading_white_space = 0
            cleaned_result_v = {
                "buggy": "\n".join([line[leading_white_space:] for line in lines])
            }
            lines = v["context"]["prefix"].splitlines()
            cleaned_result_v["prefix"] = "\n".join(
                [line[leading_white_space:] for line in lines]
            )
            lines = v["context"]["suffix"].splitlines()
            cleaned_result_v["suffix"] = "\n".join(
                [line[leading_white_space:] for line in lines]
            )
            lines = v["context"]["fix"].splitlines()
            # leading_white_space = len(lines[0]) - len(lines[0].lstrip())
            leading_white_space = 0
            cleaned_result_v["fix"] = "\n".join(
                [line[leading_white_space:] for line in lines]
            )
            buggy_line = remove_suffix(
                remove_prefix(cleaned_result_v["buggy"], cleaned_result_v["prefix"]),
                cleaned_result_v["suffix"],
            ).replace("\n", "")
            cleaned_result_v["buggy_line"] = buggy_line
            cleaned_result_v["tokens"] = v["tokens"]
            cleaned_result_v["place_num"] = len(v_s)
            cleaned_result[
                k + "-place-" + str(v_s.index(v)) + ".java"
            ] = cleaned_result_v

    result = {k: v for k, v in cleaned_result.items() if check_d4j_2(k, True)}
    print(len(result))
    return result


def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[: -len(suffix)]
    return input_string


def remove_prefix(input_string, prefix):
    if prefix and input_string.startswith(prefix):
        return input_string[len(prefix) :]
    return input_string
