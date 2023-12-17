import argparse
import json
import os
import random
import re
import time

import numpy as np
import torch

from dataset import (
    get_unified_diff,
    parse_d4j_2_full_with_repo_rare_token,
    parse_d4j_12_full_with_repo_rare_token,
)
from model import SpanLM
from template import generate_match_template


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _write_to_file(patch_file, folder, patch):
    try:
        with open(folder + "/" + patch_file, "w") as f:
            f.write(patch)
    except:
        with open(folder + "/" + patch_file, "w") as f:
            f.write("write error ... ")
        return False


def generate_templates(
    prefix: str, suffix: str, buggy_line: str, model: SpanLM, tokens, args
):
    templates = []
    leading_white_space = len(buggy_line) - len(buggy_line.lstrip())
    buggy_line = buggy_line.lstrip()

    prefix += "\n" + " " * leading_white_space
    common_prefix = []
    if args.instruction is not None:
        if "original" not in tokens:
            common_prefix.append(prefix)
        else:
            for var in tokens["original"][: args.top]:
                common_prefix.append(
                    prefix
                    + "/* use {} in the next line: */".format(
                        list(var.values())[0]["identifier"]
                    )
                )

    if len(common_prefix) == 0:
        common_prefix.append(prefix)

    # entire line replace
    for prefix in common_prefix:
        templates.append(
            (
                "lr",
                "{}\n{}".format(prefix, " " * leading_white_space),
                "\n{}".format(suffix),
                "",
                "",
            )
        )

        if len(buggy_line) > 0:
            # partial before
            str_builder = ""
            tokens = model.encode(buggy_line)
            for s in tokens:
                str_builder += model.decode(s)
                templates.append(
                    (
                        "pb",
                        "{}\n{}".format(
                            prefix, " " * leading_white_space + str_builder
                        ),
                        "\n{}".format(suffix),
                        str_builder,
                        "",
                    )
                )

            # partial after
            str_builder = ""
            for s in reversed(tokens):
                str_builder = model.decode(s) + str_builder
                templates.append(
                    (
                        "pa",
                        "{}\n{}".format(prefix, " " * leading_white_space),
                        "{}\n{}".format(str_builder, suffix),
                        "",
                        str_builder,
                    )
                )

            template_match = generate_match_template(buggy_line)
            for t_prefix, t_suffix in template_match:
                templates.append(
                    (
                        "tm",
                        "{}\n{}".format(prefix, " " * leading_white_space + t_prefix),
                        "{}\n{}".format(t_suffix, suffix),
                        t_prefix,
                        t_suffix,
                    )
                )
    return templates


def remove_comments(string):
    pattern = r"(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)"
    # first group captures quoted strings (double or single)
    # second group captures comments (//single-line or /* multi-line */)
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)

    def _replacer(match):
        # if the 2nd group (capturing comments) is not None,
        # it means we have captured a non-quoted (real) comment string.
        if match.group(2) is not None:
            return ""  # so we will return empty to remove the comment
        else:  # otherwise, we will return the 1st group
            return match.group(1)  # captured quoted-string

    return regex.sub(_replacer, string)


def suffix_repair_loop(args, model: SpanLM, prefix, suffix, file_name, folder, bug):
    start = time.time()
    repair_result = []
    p_diff = {}
    print("Repairing bug {} ... ".format(file_name.split(".")[0]))
    real_prefix = prefix
    real_suffix = suffix
    templates = generate_templates(
        prefix, suffix, bug["buggy_line"], model, bug["tokens"], args
    )

    if not model.check_size(prefix=prefix, suffix=suffix):
        return 0, []

    total_times = 0
    for template_name, prefix, suffix, t_prefix, t_suffix in templates:
        print("{}{}{}".format(prefix, ">>> [INSERT] <<<", suffix))
        chances = int((5000 / bug["place_num"]) / len(templates))
        while chances > 0:
            total_times += 1
            torch.cuda.empty_cache()
            print("Try :{}".format(total_times))
            outputs, entropies = model.predict(
                prefix=prefix, suffix=suffix, num_samples=chances
            )
            # return
            chances -= args.batch_size
            for index, output in enumerate(outputs):
                output = (
                    real_prefix
                    + "\n"
                    + t_prefix
                    + output
                    + t_suffix
                    + "\n"
                    + real_suffix
                )
                unique_output = (
                    remove_comments(output)
                    .replace(" ", "")
                    .replace("\n", "")
                    .replace("\t", "")
                    .replace("\r", "")
                )
                diff = get_unified_diff(bug["buggy"], output)
                if unique_output in p_diff:
                    repair_result[p_diff[unique_output]]["num"] += 1
                    continue
                p_diff[unique_output] = len(repair_result)
                print(diff)
                _write_to_file(
                    file_name.split(".")[0]
                    + "_"
                    + str(len(repair_result))
                    + "."
                    + file_name.split(".")[1],
                    folder,
                    output,
                )
                repair_result.append(
                    {
                        "output": output,
                        "diff": diff,
                        "finish_reason": "stop",
                        "entropy": entropies[index],
                        "valid": False,
                        "type": template_name,
                        "num": 1,
                    }
                )
    end = time.time()
    print("{} Unique Patches Generated in {}s".format(len(repair_result), end - start))

    return total_times, repair_result


def repair(args, model: SpanLM, bugs: dict):
    if not os.path.exists(args.folder):
        os.makedirs(args.folder)
    with open(args.folder + "/args.txt", "w") as f:
        f.write(str(args))

    result = {}
    t_generated = 0
    t_unique = 0
    start_t = time.time()

    # times = 0

    for file_name, bug in bugs.items():
        # if times > 0:
        #     return
        # times += 1
        prefix = bug["prefix"]
        suffix = bug["suffix"]
        n_generated, result[file_name] = suffix_repair_loop(
            args, model, prefix, suffix, file_name, args.folder, bug
        )
        if n_generated >= 1:
            t_generated += n_generated * args.batch_size
            t_unique += len(result[file_name])

        with open(args.folder + "/lm_repair.json", "w") as f:  # write to file
            json.dump(result, f)

    # return

    end_t = time.time()

    with open(args.folder + "/stats.txt", "w") as f:
        f.write("Total generated: {}\n".format(t_generated))
        f.write("Total unique: {}\n".format(t_unique))
        f.write("Total time: {}\n".format(end_t - start_t))

    with open(args.folder + "/lm_repair.json", "w") as f:  # write to file
        json.dump(result, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Salesforce/codet5-large")
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="Dataset to use, current support: defects4j-1.2, defects4j-2.0",
    )
    parser.add_argument("--project_identifier", type=str, default=None)
    parser.add_argument("--folder", type=str, default="Results/test")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--instruction", action="store_true")
    parser.add_argument("--top", type=int, default=5)

    args = parser.parse_args()

    print(args)

    if args.dataset == "defects4j-1.2":
        bugs = parse_d4j_12_full_with_repo_rare_token(args.project_identifier)
        args.language = "java"
    elif args.dataset == "defects4j-2.0":
        bugs = parse_d4j_2_full_with_repo_rare_token(args.project_identifier)
        args.language = "java"
    else:
        raise NotImplementedError("Dataset not supported")

    model = SpanLM(pretrained=args.model_name, batch_size=args.batch_size)
    repair(args, model, bugs)


if __name__ == "__main__":
    main()
