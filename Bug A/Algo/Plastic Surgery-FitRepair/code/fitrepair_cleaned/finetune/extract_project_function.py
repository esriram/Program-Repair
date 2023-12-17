import argparse
import csv
import hashlib
import os
import subprocess
import sys

import pandas as pd
from docopt import docopt
from dpu_utils.codeutils.deduplication import DuplicateDetector
from dpu_utils.utils import RichPath, run_and_debug
from tqdm import tqdm
from tree_sitter import Language, Parser
from util.pkldf2jsonl import chunked_save_df_to_jsonl

sys.path.append("function_parser/function_parser")
from language_data import LANGUAGE_METADATA
from process import DataProcessor

project_identifier_list = [
    "Chart",
    "Cli",
    "Closure",
    "Codec",
    "Collections",
    "Compress",
    "Csv",
    "Gson",
    "JacksonCore",
    "JacksonDatabind",
    "JacksonXml",
    "Jsoup",
    "JxPath",
    "Lang",
    "Math",
    "Mockito",
    "Time",
]

Identifier2Project = {
    "Chart": "jfreechart",
    "Cli": "commons-cli",
    "Closure": "closure-compiler",
    "Codec": "commons-codec",
    "Collections": "commons-collections",
    "Compress": "commons-compress",
    "Csv": "commons-csv",
    "Gson": "gson",
    "JacksonCore": "jackson-core",
    "JacksonDatabind": "jackson-databind",
    "JacksonXml": "jackson-dataformat-xml",
    "Jsoup": "jsoup",
    "JxPath": "commons-jxpath",
    "Lang": "commons-lang",
    "Math": "commons-math",
    "Mockito": "mockito",
    "Time": "joda-time",
}

extract_methods = [
    "function",
    "block",
    "function_with_bug",
    "function_with_bug_no_test_file",
    "block_with_bug",
    "function_without_bug",
    "block_without_bug",
]


def remove_duplicate_code_df(df: pd.DataFrame) -> pd.DataFrame:
    "Resolve near duplicates based upon function_tokens field in data."
    assert (
        "function_tokens" in df.columns.values
    ), "Data must contain field function_tokens"
    assert "language" in df.columns.values, "Data must contain field language"
    df.reset_index(inplace=True, drop=True)
    df["doc_id"] = df.index.values
    dd = DuplicateDetector(min_num_tokens_per_document=10)
    filter_mask = df.apply(
        lambda x: dd.add_file(
            id=x.doc_id, tokens=x.function_tokens, language=x.language
        ),
        axis=1,
    )
    # compute fuzzy duplicates
    exclusion_set = dd.compute_ids_to_exclude()
    # compute pandas.series of type boolean which flags whether or not code should be discarded
    # in order to resolve duplicates (discards all but one in each set of duplicate functions)
    exclusion_mask = df["doc_id"].apply(lambda x: x not in exclusion_set)

    # filter the data
    print(
        f"Removed {sum(~(filter_mask & exclusion_mask)):,} fuzzy duplicates out of {df.shape[0]:,} rows."
    )
    return df[filter_mask & exclusion_mask]


def label_folds(
    df: pd.DataFrame,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    holdout_ratio: float,
) -> pd.DataFrame:
    "Adds a partition column to DataFrame with values: {train, valid, test, holdout}."
    assert (
        abs(train_ratio + valid_ratio + test_ratio + holdout_ratio - 1) < 1e-5
    ), "Ratios must sum up to 1."
    # code in the same file will always go to the same split
    df["hash_key"] = df.apply(lambda x: f"{x.nwo}:{x.path}", axis=1)
    df["hash_val"] = df["hash_key"].apply(
        lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % (2**16)
    )

    train_bound = int(2**16 * train_ratio)
    valid_bound = train_bound + int(2**16 * valid_ratio)
    test_bound = valid_bound + int(2**16 * test_ratio)

    def label_splits(hash_val: int) -> str:
        if hash_val <= train_bound:
            return "train"
        elif hash_val <= valid_bound:
            return "valid"
        elif hash_val <= test_bound:
            return "test"
        else:
            return "holdout"

    # apply partition logic
    df["partition"] = df["hash_val"].apply(lambda x: label_splits(x))
    # display summary statistics
    counts = df.groupby("partition")["nwo"].count().rename("count")
    summary_df = pd.concat([counts, (counts / counts.sum()).rename("pct")], axis=1)
    print(summary_df)

    return df


def main(
    data_path,
    dataset_name,
    extract_type,
    data_duplication,
    language,
    project_identifier,
    bug_id,
):
    if dataset_name == "DefextsKotlin":
        csvPath = "/home/Project/dl/defexts/dataset-kotlin/references.csv"
        with open(csvPath) as csvData:
            fields = [
                "id",
                "url",
                "project",
                "hash",
                "commit_url",
                "build_system",
                "android",
            ]  # Must be updated anytime the references.csv is updated
            csvReader = csv.DictReader(csvData, fieldnames=fields)
            csvDict = {}
            for row in csvReader:
                csvDict[row["id"].strip()] = row
        for bug in csvDict:
            if "-".join(bug.split("-")[:-1]) in [
                "what-day-bot",
                "Simple-MsgPack",
                "zzp-matcher",
                "birthday-greetings-kata-kotlin",
                "seven-wonders",
                "thrifty",
                "platform",
                "TestArtifacts",
                "kotlinpoet",
                "okio",
                "ilias-downloader-cli",
                "patchtools",
                "Scapes",
                "UltimateTTT",
            ]:  # bug in java or groovy
                continue
            if bug in ["gradle-play-publisher-2"]:
                continue
            DataProcessor.PARSER.set_language(
                Language("util/py-tree-sitter-kotlin.so", language)
            )
            processor = DataProcessor(
                language=language,
                language_parser=LANGUAGE_METADATA[language]["language_parser"],
            )
            definitions = processor.process_dee_local_for_DefextsKotlin(
                bug, ext=LANGUAGE_METADATA[language]["ext"], extract_type=extract_type
            )
            df = pd.DataFrame(definitions)
            print(df)

            SAVE_PATH = os.path.join(data_path, dataset_name, bug, extract_type)
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            df = df.sample(frac=1, random_state=20181026)  # shuffle order of files
            data_num = len(df)

            train_ratio_ = 1.0
            valid_ratio_ = 0
            test_ratio_ = 0
            holdout_ratio_ = 0
            df = label_folds(
                df,
                train_ratio=train_ratio_,
                valid_ratio=valid_ratio_,
                test_ratio=test_ratio_,
                holdout_ratio=holdout_ratio_,
            )
            splits = ["train"]
            splits_ratio = [train_ratio_, valid_ratio_, test_ratio_, holdout_ratio_]

            output_folder = RichPath.create(SAVE_PATH)

            for split in splits:
                split_df = df[df.partition == split]

                # save dataframes as chunked jsonl files
                if not os.path.exists(os.path.join(SAVE_PATH, f"jsonl/{split}")):
                    os.makedirs(os.path.join(SAVE_PATH, f"jsonl/{split}"))
                jsonl_save_folder = output_folder.join(f"jsonl/{split}")
                print(f"Uploading data to {str(jsonl_save_folder)}")
                chunked_save_df_to_jsonl(split_df, jsonl_save_folder, num_chunks=1)
                split_data_num = int(data_num * splits_ratio[splits.index(split)])
                with open(
                    os.path.join(SAVE_PATH, f"jsonl/{split}", "len"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(str(split_data_num))
                f.close()

        return

    if "defects4j" in dataset_name:
        if project_identifier not in project_identifier_list:
            print("Project {} does not exist!".format(project_identifier))
            print("Available projects:", project_identifier_list)
            return

        dependee = Identifier2Project[project_identifier]
        bug_location_path = os.path.join(data_path, dataset_name, "location")

        # bug2SHA_output = subprocess.getoutput("defects4j query -p " + project_identifier + " -q \"revision.id.buggy\"")
        with open(
            "/home/Project/dl/AlphaRepair_finetune/defects4j/framework/projects/{}/commit-db".format(
                project_identifier
            )
        ) as commit_db_file:
            bug2SHA_output = commit_db_file.read()
        bug2SHA_list = bug2SHA_output.split("\n")
        bug2SHA = {}
        SHA_list = []
        bug_id_list = []
        for i in bug2SHA_list:
            if len(i) > 0:
                bug2SHA[i.split(",")[0]] = i.split(",")[1]
                SHA_list.append(i.split(",")[1])
                bug_id_list.append(i.split(",")[0])

        if extract_type not in extract_methods:
            print("Extract method {} does not exist!".format(extract_type))
            return
        else:
            DataProcessor.PARSER.set_language(
                Language("util/py-tree-sitter-languages.so", language)
            )
            processor = DataProcessor(
                language=language,
                language_parser=LANGUAGE_METADATA[language]["language_parser"],
            )
            if bug_id in bug_id_list:
                definitions = processor.process_dee_local_with_SHA(
                    dependee,
                    ext=LANGUAGE_METADATA[language]["ext"],
                    SHA=bug2SHA[bug_id],
                    bug2SHA=bug2SHA,
                    Identifier2Project=Identifier2Project,
                    bug_location_path=bug_location_path,
                    extract_type=extract_type,
                )
            elif bug_id == "oldest":
                definitions, oldest_sha = processor.process_dee_local_oldest(
                    dependee,
                    ext=LANGUAGE_METADATA[language]["ext"],
                    sha_list=SHA_list,
                    bug2SHA=bug2SHA,
                    Identifier2Project=Identifier2Project,
                    bug_location_path=bug_location_path,
                    extract_type=extract_type,
                )
            else:
                print("BUG {} does not exist!".format(bug_id))
                return
            df = pd.DataFrame(definitions)

    if data_duplication == "False":
        SAVE_PATH = os.path.join(
            data_path, dataset_name, dependee, extract_type + "_no_duplication", bug_id
        )
    else:
        SAVE_PATH = os.path.join(
            data_path, dataset_name, dependee, extract_type, bug_id
        )

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    if data_duplication == "False":
        df = remove_duplicate_code_df(df)
    df = df.sample(frac=1, random_state=20181026)  # shuffle order of files
    data_num = len(df)

    train_ratio_ = 1.0
    valid_ratio_ = 0
    test_ratio_ = 0
    holdout_ratio_ = 0
    df = label_folds(
        df,
        train_ratio=train_ratio_,
        valid_ratio=valid_ratio_,
        test_ratio=test_ratio_,
        holdout_ratio=holdout_ratio_,
    )
    splits = ["train"]
    splits_ratio = [train_ratio_, valid_ratio_, test_ratio_, holdout_ratio_]

    output_folder = RichPath.create(SAVE_PATH)

    for split in splits:
        split_df = df[df.partition == split]

        # save dataframes as chunked jsonl files
        if not os.path.exists(os.path.join(SAVE_PATH, f"jsonl/{split}")):
            os.makedirs(os.path.join(SAVE_PATH, f"jsonl/{split}"))
        jsonl_save_folder = output_folder.join(f"jsonl/{split}")
        print(f"Uploading data to {str(jsonl_save_folder)}")
        chunked_save_df_to_jsonl(split_df, jsonl_save_folder, num_chunks=1)
        split_data_num = int(data_num * splits_ratio[splits.index(split)])
        with open(
            os.path.join(SAVE_PATH, f"jsonl/{split}", "len"), "w", encoding="utf-8"
        ) as f:
            f.write(str(split_data_num))
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_identifier", type=str, default="Closure")
    parser.add_argument("--extract_type", type=str, default="function")
    parser.add_argument(
        "--bug_id",
        type=str,
        default="oldest",
        help='Options: one true bug id (get repo based on this bug\'s SHA) and "oldest" (get repo based on the oldest SHA)',
    ),
    parser.add_argument("--language", type=str, default="java")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/data/AlphaRepair_finetune",
        help="Path for saving dataset for finetune",
    )
    parser.add_argument("--dataset_name", type=str, default="defects4j")
    parser.add_argument("--data_type", type=str, default="NL+PL")
    parser.add_argument("--data_duplication", type=str, default="True")
    args = parser.parse_args()
    main(
        data_path=args.data_path,
        dataset_name=args.dataset_name,
        extract_type=args.extract_type,
        data_duplication=args.data_duplication,
        language=args.language,
        project_identifier=args.project_identifier,
        bug_id=args.bug_id,
    )
