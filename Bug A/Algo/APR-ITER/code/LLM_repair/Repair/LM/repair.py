import argparse
import sys
import torch
import os
import json
import time

sys.path.append(os.path.dirname(os.path.join(sys.path[0], '../../')))  # Hack
sys.path.append(os.path.dirname(os.path.join(sys.path[0], '../../Dataset/')))

from model import GPT2, SpanLM
from Dataset.parse_quixbugs import parse_python, get_unified_diff, parse_java, parse_java_single_line
from Dataset.parse_d4j import clean_parse_d4j, clean_parse_d4j_single_hunk, clean_parse_d4j_single_line
from Dataset.parse_manybugs import clean_parse_manybugs, clean_parse_manybugs_single_hunk, clean_parse_manybugs_single_line
from Repair.prompt import JAVA_LONG_VARY_PROMPT, VARY_BASE_PROMPT, C_VARY_PROMPT
from Repair.util import pick_smallest_example_fix, set_seed, _run_validation


def suffix_repair_loop(args, model: SpanLM, prefix, suffix, file_name, folder, bug, t_chances, skip_val=True):
    start = time.time()
    repair_result = []
    p_diff = {}
    print("Repairing bug {} ... ".format(file_name.split(".")[0]))
    print(prefix)
    print(">>> [INSERT] <<<")
    print(suffix)
    if not model.check_input(prefix, suffix, bug['buggy']):
        return 0, False, False, repair_result

    total_times = 0
    while t_chances > 0:
        total_times += 1
        torch.cuda.empty_cache()
        print("Try :{}".format(total_times))
        well, early_stop, outputs, entropies = model.model_predict(prefix=prefix, suffix=suffix,
                                                                   do_sample=True,
                                                                   buggy=bug['buggy'],
                                                                   num_samples=t_chances)
        t_chances -= args.batch_size
        if well:
            for index, output in enumerate(outputs):
                output = prefix + output + suffix
                diff = get_unified_diff(bug['buggy'], output)
                if diff in p_diff:
                    repair_result[p_diff[diff]]['num'] += 1
                    continue
                p_diff[diff] = len(repair_result)
                print(diff)
                repair_result.append({'output': output,
                                      'diff': diff,
                                      'finish_reason': 'stop',
                                      'entropy': entropies[index],
                                      'valid': _run_validation(file_name.split(".")[0],
                                                               file_name.split(".")[0] + "_" + str(
                                                                   len(repair_result)) + "." + file_name.split(".")[1],
                                                               folder, output, skip_val=skip_val),
                                      'num': 1})
    end = time.time()
    print("{} Unique Patches Generated in {}s".format(len(repair_result), end - start))

    return total_times, False, False, repair_result


def single_line_repair_loop(args, model, prefix, suffix, file_name, folder, bug, t_chances, skip_val=True):
    start = time.time()
    repair_result = []
    p_diff = {}
    print("Repairing bug {} ... ".format(file_name.split(".")[0]))
    print(prefix)
    if not model.check_input(prefix, ""):
        return 0, False, False, repair_result

    total_times = 0
    while t_chances > 0:
        total_times += 1
        torch.cuda.empty_cache()
        print("Try :{}".format(total_times))
        well, length, outputs, entropies = model.model_predict(prefix, bug['buggy'], do_sample=True,
                                                               num_samples=t_chances)
        t_chances -= args.batch_size
        if well:
            for index, output in enumerate(outputs):
                output = prefix + output + suffix
                diff = get_unified_diff(bug['buggy'], output)
                if diff in p_diff:
                    repair_result[p_diff[diff]]['num'] += 1
                    continue
                p_diff[diff] = len(repair_result)
                print(diff)
                repair_result.append({'output': output,
                    'diff': diff,
                    'finish_reason': 'stop',
                    'entropy': entropies[index],
                    'valid': _run_validation(file_name.split(".")[0],
                                             file_name.split(".")[0] + "_" + str(
                                                 len(repair_result)) + "." + file_name.split(".")[1],
                                             folder, output, skip_val=skip_val),
                    'num': 1})

    end = time.time()

    print("{} Unique Patches Generated in {}s".format(len(repair_result), end - start))

    return len(repair_result), False, False, repair_result


def repair_loop(args, model, prompt, file_name, folder, bug, t_chances, skip_val=True):
    start = time.time()
    repair_result = []
    p_diff = {}
    print("Repairing bug {} ... ".format(file_name.split(".")[0]))
    print(prompt)
    if not model.check_input(prompt, bug['buggy']):
        return 0, False, False, repair_result

    total_times = 0
    while t_chances > 0:
        total_times += 1
        torch.cuda.empty_cache()
        print("Try :{}".format(total_times))
        well, length, outputs, entropies = model.model_predict(prompt, bug['buggy'], do_sample=True,
                                                               num_samples=t_chances)
        t_chances -= args.batch_size
        if well:
            for index, output in enumerate(outputs):
                diff = get_unified_diff(bug['buggy'], output)
                if diff in p_diff:
                    repair_result[p_diff[diff]]['num'] += 1
                    continue
                p_diff[diff] = len(repair_result)
                print(diff)
                repair_result.append({'output': output,
                                      'diff': diff,
                                      'finish_reason': 'stop',
                                      'entropy': entropies[index],
                                      'valid': _run_validation(file_name.split(".")[0],
                                                               file_name.split(".")[0] + "_" + str(
                                                                   len(repair_result)) + "." + file_name.split(".")[1],
                                                               folder, output, skip_val=skip_val),
                                      'num': 1})

    end = time.time()

    print("{} Unique Patches Generated in {}s".format(len(repair_result), end - start))

    return len(repair_result), False, False, repair_result


def suffix_repair(args, model, bugs, folder, chances, skip_val=True):
    """
    Suffix LM repair loop
    :param args: input arguments
    :param model: model to use for repair
    :param bugs: dict of bugs
    :param folder: folder to save the files
    :param chances: number of chances to try to repair
    :param skip_val: if True, skip validation
    :param set_suffix: set prefix for infilling
    :param set_prefix: set suffix for infilling
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(folder + "/args.txt", "w") as f:
        f.write(str(args))

    result = {}
    t_generated = 0
    t_unique = 0
    start_t = time.time()
    for file_name, bug in bugs.items():
        if 'suffix' not in bug:
            continue
        suffix = "\n" + bug['suffix']
        # leading white space removal is needed to help with codet5 prediction since it does not have concept of
        # white spaces
        # leading_white_space = len(bug['buggy'].splitlines()[bug['line_no']]) - len(bug['buggy'].splitlines()[bug['line_no']].lstrip())
        prefix = bug['prefix'] + "\n" #+ " "*leading_white_space
        n_generated, valid, first_try, result[file_name] = suffix_repair_loop(args, model, prefix, suffix,
                                                                                  file_name,
                                                                                  folder, bug,
                                                                                  chances, skip_val)
        if n_generated >= 1:
            t_generated += chances
            t_unique += len(result[file_name])

    end_t = time.time()

    with open(folder + "/stats.txt", "w") as f:
        f.write("Total generated: {}\n".format(t_generated))
        f.write("Total unique: {}\n".format(t_unique))
        f.write("Total time: {}\n".format(end_t - start_t))

    with open(folder + "/lm_repair.json", "w") as f:  # write to file
        json.dump(result, f)


def single_line_repair(args, model, bugs, folder, chances, skip_val):
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(folder + "/args.txt", "w") as f:
        f.write(str(args))

    result = {}
    t_generated = 0
    t_unique = 0
    start_t = time.time()
    for file_name, bug in bugs.items():
        if "suffix" not in bug:
            continue

        suffix = "\n" + bug['suffix']
        prefix = bug['prefix'] + "\n"
        n_generated, valid, first_try, result[file_name] = single_line_repair_loop(args, model, prefix, suffix, file_name, folder, bug,
                                                                                   chances, skip_val)
        if n_generated >= 1:
            t_generated += chances
            t_unique += len(result[file_name])

    end_t = time.time()

    with open(folder + "/stats.txt", "w") as f:
        f.write("Total generated: {}\n".format(t_generated))
        f.write("Total unique: {}\n".format(t_unique))
        f.write("Total time: {}\n".format(end_t - start_t))

    with open(folder + "/lm_repair.json", "w") as f:  # write to file
        json.dump(result, f)


def repair(args, model, bugs, folder, used_prompt, chances, skip_val=True, only_same=True):
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(folder + "/prompt.txt", "w") as f:
        f.write(used_prompt)
    with open(folder + "/args.txt", "w") as f:
        f.write(str(args))

    result = {}
    t_generated = 0
    t_unique = 0
    start_t = time.time()
    for file_name, bug in bugs.items():
        if "Collections" in file_name:
            example_bug, example_fix = pick_smallest_example_fix(bugs, file_name, only_same=False)
        else:
            example_bug, example_fix = pick_smallest_example_fix(bugs, file_name, only_same=only_same)
        prompt = used_prompt.format(example_bug=example_bug, example_fix=example_fix, bug=bug['buggy'])
        n_generated, valid, first_try, result[file_name] = repair_loop(args, model, prompt, file_name, folder, bug,
                                                                       chances, skip_val)
        if n_generated >= 1:
            t_generated += chances
            t_unique += len(result[file_name])

    end_t = time.time()

    with open(folder + "/stats.txt", "w") as f:
        f.write("Total generated: {}\n".format(t_generated))
        f.write("Total unique: {}\n".format(t_unique))
        f.write("Total time: {}\n".format(end_t - start_t))

    with open(folder + "/lm_repair.json", "w") as f:  # write to file
        json.dump(result, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="defects4j",
                        help="Dataset to use, current support: defects4j, quixbug-python, quixbugs-java, manybugs")
    parser.add_argument("--chances", type=int, default=1)
    parser.add_argument("--skip_val", action="store_true", default=False)
    parser.add_argument("--folder", type=str, default="Results/test")
    parser.add_argument("--weight", type=str, default=None)
    parser.add_argument("--suffix", action="store_true", default=False)
    parser.add_argument("--single_line", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=420)
    args = parser.parse_args()
    if args.dataset == "defects4j":
        if args.suffix:
            dataset = clean_parse_d4j_single_hunk(folder="../../")
        elif args.single_line:
            dataset = clean_parse_d4j_single_line(folder="../../")
        else:
            dataset = clean_parse_d4j(folder="../../")
        prompt = JAVA_LONG_VARY_PROMPT
        stop = "// Provide a fix for the buggy function"
        if args.single_line:
            stop = "\n"
        args.language = "java"
    elif args.dataset == "quixbug-python":
        dataset = parse_python(folder='../../')
        prompt = VARY_BASE_PROMPT
        stop = "# Provide a fix for the buggy function"
        if args.single_line:
            stop = "\n"
        args.language = "python"
    elif args.dataset == "quixbug-java":
        if args.single_line:
            dataset = parse_java_single_line(folder="../../")
        else:
            dataset = parse_java(folder='../../')
        prompt = JAVA_LONG_VARY_PROMPT
        stop = "// Provide a fix for the buggy function"
        if args.single_line:
            stop = "\n"
        args.language = "java"
    elif args.dataset == "manybugs":
        if args.single_line:
            dataset = clean_parse_manybugs_single_line(folder='../../')
        elif args.suffix:
            dataset = clean_parse_manybugs_single_hunk(folder='../../')
        else:
            dataset = clean_parse_manybugs(folder='../../')
        prompt = C_VARY_PROMPT
        stop = "/* Provide a fix for the buggy function */"
        if args.single_line:
            stop = "\n"
        args.language = "c"
    else:
        print("Unknown dataset: {}".format(args.dataset))
        return -1

    set_seed(args.seed)
    if args.suffix:
        model = SpanLM(pretrained=args.model_name, weight=args.weight, batch_size=args.batch_size)
        suffix_repair(args, model, dataset, args.folder, args.chances, args.skip_val)
    elif args.single_line:
        model = GPT2(batch_size=args.batch_size, pretrained=args.model_name, stop=stop, weight=args.weight)
        single_line_repair(args, model, dataset, args.folder, args.chances, args.skip_val)
    else:
        model = GPT2(batch_size=args.batch_size, pretrained=args.model_name, stop=stop, weight=args.weight)
        repair(args, model, dataset, args.folder, prompt, args.chances, args.skip_val, only_same=args.dataset.startswith("defects4j"))


if __name__ == '__main__':
    main()
