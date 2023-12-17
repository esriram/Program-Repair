import glob
import itertools
import json
import math
import os
import pickle

# import tokenizations
import random
import re
import sys
import time
from pathlib import Path

import jellyfish
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
from transformers import RobertaTokenizer
from tree_sitter import Language, Parser

# from fuzzysearch import find_near_matches
# from fuzzywuzzy import process
# import Levenshtein

sys.path.append("../")
# from util.tree_sitter_processor import TreeSitterProcessor
from util.util import (
    check_token_length,
    collate_2d,
    get_index,
    handle_special_cases_for_mask_code_lines,
    merge_intervals,
    pad_to_len,
    to_device,
)

supported_repo_names = [
    "jfreechart",
    "commons-cli",
    "closure-compiler",
    "commons-codec",
    "commons-collections",
    "commons-compress",
    "commons-csv",
    "gson",
    "jackson-core",
    "jackson-databind",
    "jackson-dataformat-xml",
    "jsoup",
    "commons-jxpath",
    "commons-lang",
    "commons-math",
    "mockito",
    "joda-time",
]


class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


# Basic MLM dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, pretraining_objective):
        # store encodings internally
        self.encodings = encodings
        self.pretraining_objective = pretraining_objective

    def __len__(self):
        # return the number of samples
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        if "MLM" in self.pretraining_objective:
            return self.encodings["input_ids"][i], self.encodings["labels"][i]


def _build_mask_index(
    tokenizer,
    codes_tokens: list,
    codes_original: list,
    docstring_tokens: list,
    masking_rate: float,
    masking_style: int,
) -> torch.tensor:
    codes_tokens_flags = []
    docstring_tokens_flags = []
    # only using one <MASK> token but only on one line as well
    if masking_style == 9:
        for original_code in codes_original:
            lines = original_code.splitlines()
            if len(lines) <= 1:
                continue
            if (
                len(
                    tokenizer(original_code, max_length=512, truncation=False)[
                        "input_ids"
                    ]
                )
                >= 512
            ):
                continue
            maximum = max([len(line.lstrip()) for line in lines[1:]])
            long_lines = [
                index + 1
                for index, line in enumerate(lines[1:])
                if (len(line.lstrip()) > maximum / 3)
                and not line.lstrip().startswith("//")
            ]

            if len(long_lines) > 0:  # 8377
                for choice in np.random.choice(
                    long_lines, min(20, len(long_lines)), replace=False
                ):
                    tmp_lines = lines.copy()
                    docstring_tokens_flags.append(
                        "<extra_id_0>\n" + tmp_lines[choice] + "<extra_id_1>"
                    )
                    tmp_lines[choice] = "<extra_id_0>"
                    codes_tokens_flags.append("\n".join(tmp_lines))

    elif masking_style == 10:
        entire_line = 0
        prefix = 0
        suffix = 0
        for original_code in codes_original:
            lines = original_code.splitlines()
            if len(lines) <= 1:
                continue
            maximum = max([len(line.lstrip()) for line in lines[1:]])
            long_lines = [
                index + 1
                for index, line in enumerate(lines[1:])
                if (len(line.lstrip()) > maximum / 3)
                and not line.lstrip().startswith("//")
            ]

            if len(long_lines) > 0:  # 8377
                for choice in np.random.choice(
                    long_lines, min(10, len(long_lines)), replace=False
                ):
                    tmp_lines = lines.copy()
                    buggy_prefix, span_line, label, type = generate_training_templates(
                        tokenizer, tmp_lines[choice], tmp_lines
                    )
                    docstring_tokens_flags.append(label)
                    tmp_lines[choice] = span_line
                    codes_tokens_flags.append(
                        buggy_prefix + "\n" + "\n".join(tmp_lines)
                    )
                    if type == 0:
                        entire_line += 1
                    elif type == 1:
                        prefix += 1
                    else:
                        suffix += 1
    elif masking_style == 11:
        JA_LANGUAGE = Language("../build/my-language.so", "java")
        parser = Parser()
        parser.set_language(JA_LANGUAGE)
        for original_code in codes_original:

            lines = original_code.splitlines()
            if len(lines) <= 1:
                continue
            if (
                len(
                    tokenizer(original_code, max_length=512, truncation=False)[
                        "input_ids"
                    ]
                )
                >= 512
            ):
                print("here")
                continue
            maximum = max([len(line.lstrip()) for line in lines[1:]])
            long_lines = [
                index + 1
                for index, line in enumerate(lines[1:])
                if (len(line.lstrip()) > maximum / 2)
                and not line.lstrip().startswith("//")
            ]

            if len(long_lines) > 0:  # 8377
                for choice in np.random.choice(
                    long_lines, min(10, len(long_lines)), replace=False
                ):
                    tmp_lines = lines.copy()
                    s = random.randint(0, 1)
                    if s == 0:
                        docstring_tokens_flags.append(
                            "<extra_id_0>\n" + tmp_lines[choice] + "<extra_id_1>"
                        )
                        tmp_lines[choice] = "<extra_id_0>"
                        codes_tokens_flags.append("\n".join(tmp_lines))
                    else:
                        span_line, label = generate_ast_training(
                            tmp_lines[choice], parser
                        )
                        docstring_tokens_flags.append(label)
                        tmp_lines[choice] = span_line
                        codes_tokens_flags.append("\n".join(tmp_lines))
    elif masking_style == 12:
        for original_code in codes_original:
            lines = original_code.splitlines()
            if len(lines) <= 1:
                continue
            if (
                len(
                    tokenizer(original_code, max_length=512, truncation=False)[
                        "input_ids"
                    ]
                )
                >= 512
            ):
                continue
            maximum = max([len(line.lstrip()) for line in lines[1:]])
            long_lines = [
                index + 1
                for index, line in enumerate(lines[1:])
                if (len(line.lstrip()) > maximum / 3)
                and not line.lstrip().startswith("//")
            ]

            if len(long_lines) > 0:  # 8377
                for choice in np.random.choice(
                    long_lines, min(20, len(long_lines)), replace=False
                ):
                    for label, span_line in generate_reverse_templates(
                        lines[choice], tokenizer
                    ):
                        tmp_lines = lines.copy()
                        docstring_tokens_flags.append(label)
                        tmp_lines[choice] = span_line
                        codes_tokens_flags.append("\n".join(tmp_lines))

    # print(entire_line, prefix, suffix)
    return codes_tokens_flags, docstring_tokens_flags


def subtree_masking(node, code, span_token):
    span_maskings = set()
    start = node.start_point[1]
    end = node.end_point[1]
    if (
        node.type != code[start:end].decode("utf-8")
        and code[start:end].decode("utf-8") != code.decode("utf-8")
        and node.type != "block"
        and node.type != "}"
        and node.type != ";"
    ):
        span_maskings.add(
            (
                code[:start].decode("utf-8") + span_token + code[end:].decode("utf-8"),
                code[start:end].decode("utf-8"),
            )
        )

    for child in node.children:
        span_maskings |= subtree_masking(child, code, span_token)

    return span_maskings


def match_conditional_expression(code):
    ret = []
    if re.match(r"if\s?\(.+\&\&.+\)\s?{$", code):
        s_code = code.split("&&")
        ret.append(
            (
                s_code[1].split(")")[0],
                s_code[0] + "&&" + "<extra_id_0>" + ")" + s_code[1].split(")")[-1],
            )
        )
    elif re.match(r"if\s?\(.+\|\|.+\)\s?{$", code):
        s_code = code.split("||")
        ret.append(
            (
                s_code[1].split(")")[0],
                s_code[0] + "||" + "<extra_id_0>" + ")" + s_code[1].split(")")[-1],
            )
        )

    return ret


def match_calling_function(code):
    ret = []
    matches = re.finditer(r"[^)(\s]+\([^)(]+\)", code)
    for match in matches:
        matched_code = match.group()
        sc = code.split(matched_code)
        print(matched_code)
        # assert (len(sc) == 2)
        # if len(sc) != 2:
        #     continue
        ret.append(
            (
                matched_code.split("(")[0],
                sc[0]
                + "<extra_id_0>"
                + "("
                + "".join(matched_code.split("(")[1:])
                + matched_code.join(sc[1:]),
            )
        )

    return ret


def match_function_api_call(code):  # lowest level
    ret = []
    matches = re.finditer(r"\([^)(]+\)", code)
    for match in matches:  # Match single function api print(abc)
        matched_code = match.group()
        sc = code.split(matched_code)
        if len(sc) != 2:
            continue

        t_prefix = sc[0] + "("
        t_suffix = ")" + sc[1]
        if t_prefix not in [v[0] for v in ret] and t_suffix not in [v[1] for v in ret]:
            ret.append((matched_code[1:-1], t_prefix + "<extra_id_0>" + t_suffix))

    return ret


def _match_function_multi_input_api_call_generate_template(matched_code):
    ret = []
    parameters = matched_code.split(",")

    # ret.append("(<mask>"+",<mask>"*(len(parameters)-1)+")") # Replace all variables.
    # ret.append(("(", "," + matched_code + ")")) # add variable in beginning
    # ret.append(("("+matched_code + ",", ")")) # add variable in end

    for index, parameter in enumerate(
        parameters
    ):  # Replace each variable with mask while keeping others
        new_code = "("
        for jindex in range(len(parameters)):
            add_code = "<mask>"
            if index != jindex:
                add_code = parameters[jindex]

            if jindex != 0:
                new_code += "," + add_code
            else:
                new_code += add_code
        new_code += ")"
        sc = new_code.split("<mask>")

        if len(sc) != 2:
            continue
        # print(parameter, sc[0], sc[1])
        ret.append((parameter, sc[0] + "<extra_id_0>" + sc[1]))

    return ret


def match_function_multi_input_api_call(code):  # lowest level

    ret = []
    matches = re.finditer(r"\([^)(]+,[^)(]+\)", code)
    for match in matches:  # Match single function api print(abc)
        matched_code = match.group()
        sc = code.split(matched_code)
        # assert (len(sc) == 2)
        if len(sc) != 2:
            continue
        matched_code = matched_code[1:-1]
        for label, span_line in _match_function_multi_input_api_call_generate_template(
            matched_code
        ):
            ret.append((label, sc[0] + span_line + sc[1]))

    return ret


def match_templates(code):
    ret = []
    # Not very smart templates
    ret.extend(match_conditional_expression(code))
    ret.extend(match_function_api_call(code))
    ret.extend(match_function_multi_input_api_call(code))
    ret.extend(match_calling_function(code))
    return ret


def generate_reverse_templates(g_code, tokenizer):
    leading_white_space = len(g_code) - len(g_code.lstrip())
    g_code = g_code.strip()
    templates = match_templates(g_code)
    if len(templates) != 0:
        ret = []
        for label, span_line in templates:
            ret.append(
                (
                    "<extra_id_0>" + label + "<extra_id_1>",
                    " " * leading_white_space + span_line,
                )
            )
        return ret
    while True:
        ret = []
        type = np.random.randint(0, 3)
        if type == 0:  # line replace
            mask_line = "<extra_id_0>"
            label = (
                "<extra_id_0>\n" + " " * leading_white_space + g_code + "<extra_id_1>"
            )
            ret.append((label, mask_line))
        elif type == 1:  # prefix
            tokens = tokenizer.encode(
                g_code.strip(), return_tensors="pt", add_special_tokens=False
            )[0]
            if len(tokens) == 1:
                continue
            keeps = np.random.choice(
                range(1, len(tokens)), min(5, len(range(1, len(tokens)))), replace=False
            )
            for keep in keeps:
                mask_line = " " * leading_white_space
                for s in range(keep):
                    mask_line += tokenizer.decode(tokens[s])
                mask_line += "<extra_id_0>"
                label = "<extra_id_0>"
                for s in range(keep, len(tokens)):
                    label += tokenizer.decode(tokens[s])
                label += "<extra_id_1>"
                ret.append((label, mask_line))
        elif type == 2:  # suffix
            tokens = tokenizer.encode(
                g_code.strip(), return_tensors="pt", add_special_tokens=False
            )[0]
            if len(tokens) == 1:
                continue
            keeps = np.random.choice(
                range(1, len(tokens)), min(5, len(range(1, len(tokens)))), replace=False
            )
            for keep in keeps:
                mask_line = " " * leading_white_space + "<extra_id_0>"
                for s in range(keep, len(tokens)):
                    mask_line += tokenizer.decode(tokens[s])
                label = "<extra_id_0>"
                for s in range(keep):
                    label += tokenizer.decode(tokens[s])
                label += "<extra_id_1>"
                ret.append((label, mask_line))
        return ret


def generate_ast_training(g_code, parser):
    leading_white_space = len(g_code) - len(g_code.lstrip())
    g_code = g_code.strip()
    b_code = bytes(g_code, "utf8")
    tree = parser.parse(b_code)
    root_node = tree.root_node
    maskings = list(subtree_masking(root_node, b_code, "<extra_id_0>"))
    if len(maskings) == 0:
        mask_line = "<extra_id_0>"
        label = "<extra_id_0>\n" + " " * leading_white_space + g_code + "<extra_id_1>"
    else:
        mask_line, label = random.choice(maskings)
        mask_line = " " * leading_white_space + mask_line
        label = "<extra_id_0>" + label + "<extra_id_1>"
    return mask_line, label


def generate_training_templates(tokenizer, g_code, codelines):
    b_prefix = "// buggy line: "
    type = np.random.randint(1, 3)
    leading_white_space = len(g_code) - len(g_code.lstrip())
    while True:
        if type == 0:  # line replace
            dists = []
            for l in codelines:
                if g_code.strip() == l.strip():
                    continue
                dists.append(
                    (
                        jellyfish.levenshtein_distance(g_code.strip(), l.strip()),
                        l.strip(),
                    )
                )
            dists = [x[1] for x in sorted(dists, key=lambda tup: tup[0])]
            line = np.random.choice(dists[:1], 1, replace=False)[0]
            return (
                b_prefix + line,
                "<extra_id_0>",
                "<extra_id_0>\n" + g_code + "<extra_id_1>",
                type,
            )
        elif type == 1:  # prefix
            prefixes_possible = []
            p_codelines = [x.strip() for x in codelines if x.strip() != g_code.strip()]
            tokens = tokenizer.encode(
                g_code.strip(), return_tensors="pt", add_special_tokens=False
            )[0]
            str_builder = ""
            for s in tokens[:-1]:
                str_builder += tokenizer.decode(s)
                possible = [
                    (l, str_builder) for l in p_codelines if l.startswith(str_builder)
                ]
                if len(possible) == 0:
                    break
                prefixes_possible.append(possible)
            if len(prefixes_possible) == 0:
                type = 0
                continue
            i = np.random.randint(0, len(prefixes_possible))
            j = np.random.randint(0, len(prefixes_possible[i]))
            return (
                b_prefix + prefixes_possible[i][j][0],
                " " * leading_white_space + prefixes_possible[i][j][1] + "<extra_id_0>",
                "<extra_id_0>"
                + g_code.removeprefix(
                    " " * leading_white_space + prefixes_possible[i][j][1]
                )
                + "<extra_id_1>",
                type,
            )
        elif type == 2:  # suffix
            suffixes_possible = []
            p_codelines = [x.strip() for x in codelines if x.strip() != g_code.strip()]
            tokens = tokenizer.encode(
                g_code.strip(), return_tensors="pt", add_special_tokens=False
            )[0]
            str_builder = ""
            for s in reversed(tokens)[:-1]:
                str_builder = tokenizer.decode(s) + str_builder
                possible = [
                    (l, str_builder) for l in p_codelines if l.endswith(str_builder)
                ]
                if len(possible) == 0:
                    break
                suffixes_possible.append(possible)
            if len(suffixes_possible) <= 3:
                type = 1
                continue
            i = np.random.randint(3, len(suffixes_possible))
            j = np.random.randint(0, len(suffixes_possible[i]))
            # print(b_prefix + suffixes_possible[i][j][0])
            # print(suffixes_possible[i][j][1])
            # print(g_code.removesuffix(suffixes_possible[i][j][1]).lstrip())
            return (
                b_prefix + suffixes_possible[i][j][0],
                " " * leading_white_space + "<extra_id_0>" + suffixes_possible[i][j][1],
                "<extra_id_0>"
                + g_code.removesuffix(suffixes_possible[i][j][1]).lstrip()
                + "<extra_id_1>",
                type,
            )


def _build_mask_index_PL(
    codes_tokens: list, docstring_tokens: list, masking_rate: float, masking_style: int
) -> torch.tensor:
    codes_tokens_flags = []

    if masking_style == 6:
        spans = []
        span_length = [m for m in range(1, 6)]
        p = 0.2
        len_distrib = [p for i in span_length]
        no_repeat = True
        for x in tqdm(range(len(codes_tokens))):
            mask_arr_x = 0
            codes_tokens_flag = [0 for m in range(len(codes_tokens[x]))]
            masking_sum = round(masking_rate * int(len(codes_tokens[x])))
            while mask_arr_x < masking_sum:
                span_len = np.random.choice(span_length, p=len_distrib)
                start_token_idx = np.random.randint(len(codes_tokens[x]) - span_len + 1)
                if no_repeat:
                    if (
                        1
                        in codes_tokens_flag[
                            start_token_idx : start_token_idx + span_len
                        ]
                    ):
                        continue
                else:
                    if codes_tokens_flag[start_token_idx] == 1:
                        continue
                for i in range(start_token_idx, start_token_idx + span_len):
                    codes_tokens_flag[i] = 1
                mask_arr_x += span_len
            codes_tokens_flags.append(codes_tokens_flag)

    return codes_tokens_flags


def _grab_raw_dataset_num(repo_name: str, data_path_info, dataset_type: str):
    raw_dataset_num = -1
    data_path = os.path.join(data_path_info[0], data_path_info[1])
    if repo_name not in supported_repo_names:
        raise ValueError(
            f"Repo {repo_name} is not supported. Supported repoes are {supported_repo_names}"
        )
    else:
        with open(
            "{}/{}/{}/{}/jsonl/{}/len".format(
                data_path, repo_name, data_path_info[2], data_path_info[3], dataset_type
            ),
            "r",
        ) as f:
            raw_dataset_num = int(f.read())
            print("{} Number of raw functions".format(raw_dataset_num))
    return raw_dataset_num


def _grab_raw_dataset(
    repo_name: str, data_path_info, num_of_files: int, dataset_type: str
):
    raw_dataset = []
    data_path = os.path.join(data_path_info[0], data_path_info[1])
    if repo_name not in supported_repo_names:
        raise ValueError(
            f"Repo {repo_name} is not supported. Supported repoes are {supported_repo_names}"
        )
    else:
        print(
            "{}/{}/{}/{}/jsonl/{}/*.jsonl".format(
                data_path, repo_name, data_path_info[2], data_path_info[3], dataset_type
            )
        )
        for file in glob.glob(
            "{}/{}/{}/{}/jsonl/{}/*.jsonl".format(
                data_path, repo_name, data_path_info[2], data_path_info[3], dataset_type
            )
        )[:num_of_files]:
            with open(file, "r") as f:
                raw_dataset.extend(f.readlines())
            print("grabbed data from file {}".format(file))

    print("Type: {} Number of raw functions: {}".format(dataset_type, len(raw_dataset)))
    return raw_dataset


def _grab_raw_dataset_from_DefextsKotlin(
    repo_name: str, data_path_info, num_of_files: int, dataset_type: str
):
    raw_dataset = []
    data_path = os.path.join(data_path_info[0], data_path_info[1])
    for file in glob.glob(
        "{}/{}/{}/jsonl/{}/*.jsonl".format(
            data_path, repo_name, data_path_info[2], dataset_type
        )
    )[:num_of_files]:
        with open(file, "r") as f:
            raw_dataset.extend(f.readlines())
        print("grabbed data from file {}".format(file))

    print("Type: {} Number of raw functions: {}".format(dataset_type, len(raw_dataset)))
    return raw_dataset


def build_dataset(
    tokenizer: RobertaTokenizer,
    masking_rate: float,
    masking_style: int,
    pretraining_objective: str,
    static_repeat: int,
    repo_name: str,
    data_path_info,
    device,
    seed,
) -> Dataset:
    if data_path_info[1] == "DefextsKotlin":
        raw_dataset = _grab_raw_dataset_from_DefextsKotlin(
            repo_name, data_path_info, num_of_files=1000, dataset_type="train"
        )
    else:
        raw_dataset = _grab_raw_dataset(
            repo_name, data_path_info, num_of_files=1000, dataset_type="train"
        )

    codes_tokens = []
    docstring_tokens = []
    codes_original = []
    docstrings_original = []
    include_new_line_token = False
    for x in raw_dataset:
        code_tokens = json.loads(x)["function_tokens"]
        docstring_token = json.loads(x)["docstring_tokens"]
        code_original = json.loads(x)["function"]
        docstring_original = json.loads(x)["docstring"]
        if include_new_line_token:
            code_tokens_original = code_tokens
            code_lines = []
            code_line = []
            codes_tokens_x_len = len(code_tokens_original)
            code_original_in_while = code_original
            codes_tokens_idx = -1
            while codes_tokens_x_len > 0:
                codes_tokens_idx += 1
                if (
                    code_original_in_while.find(code_tokens_original[codes_tokens_idx])
                    != 0
                ):
                    if code_original_in_while[0] == "\n":
                        code_lines.append(code_line)
                        code_line = []
                        code_original_in_while = code_original_in_while.lstrip()
                        codes_tokens_idx -= 1
                        continue
                code_line.append(code_tokens_original[codes_tokens_idx])
                code_original_in_while = code_original_in_while[
                    len(code_tokens_original[codes_tokens_idx]) :
                ]
                while len(code_original_in_while) > 0 and (
                    code_original_in_while[0] == " "
                    or code_original_in_while[0] == "\r"
                    or code_original_in_while[0] == "\t"
                ):
                    code_original_in_while = code_original_in_while[1:]
                codes_tokens_x_len -= 1
            code_lines.append(code_line)
            code_tokens_refined = []
            for line in code_lines:
                code_tokens_refined += line
                code_tokens_refined.append("\n")
            codes_tokens.append(code_tokens_refined)
        else:
            codes_tokens.append(code_tokens)
        docstring_tokens.append(docstring_token)
        docstrings_original.append(docstring_original)
        codes_original.append(code_original)

    random_show_idxs = [random.randint(0, len(codes_tokens) - 1) for _ in range(10)]
    input_ids_list, labels_list = _build_mask_index(
        tokenizer,
        codes_tokens,
        codes_original,
        docstring_tokens,
        masking_rate,
        masking_style,
    )
    for x in range(len(input_ids_list)):
        if x in random_show_idxs:
            print("Example", x)
            print("Input_ids:", input_ids_list[x])
            print("Labels:", labels_list[x])
            print("*" * 10)
    labels = tokenizer(
        labels_list,
        return_tensors="pt",
        max_length=512,
        padding="max_length",
        truncation=True,
        add_special_tokens=False,
    ).input_ids
    input_ids = tokenizer(
        input_ids_list,
        return_tensors="pt",
        max_length=512,
        padding="max_length",
        truncation=True,
    ).input_ids
    encodings = {"input_ids": input_ids, "labels": labels}
    dataset = Dataset(encodings, pretraining_objective)
    return dataset
