"""
Usage:
    process.py [options] INPUT_DIR OUTPUT_DIR

Options:
    -h --help
    --language LANGUAGE             Language
    --processes PROCESSES           # of processes to use [default: 16]
    --license-filter FILE           License metadata to filter, every row contains [nwo, license, language, score] (e.g. ['pandas-dev/pandas', 'bsd-3-clause', 'Python', 0.9997])
    --tree-sitter-build FILE        [default: /src/build/py-tree-sitter-languages.so]
"""
import functools
import os
import pickle
from multiprocessing import Pool
from os import PathLike
from typing import Any, Dict, List, Optional, Tuple, Type

import kopyt
import pandas as pd
import tokenizations
from docopt import docopt
from dpu_utils.codeutils.deduplication import DuplicateDetector
from language_data import LANGUAGE_METADATA
from parsers.language_parser import LanguageParser, tokenize_docstring
from tree_sitter import Language, Parser
from treelib import Node, Tree
from utils import (
    delete_multiple_element,
    download,
    download_local,
    download_local_for_DefextsKotlin,
    download_local_svn_with_SHA,
    flatten,
    get_oldest_sha,
    get_oldest_sha_svn,
    get_sha,
    get_sha_svn,
    go_to_sha,
    remap_nwo,
    walk,
)


class DataProcessor:

    PARSER = Parser()

    def __init__(self, language: str, language_parser: Type[LanguageParser]):
        self.language = language
        self.language_parser = language_parser
        self.tree_v = Tree()
        # self.parser = Parser()
        # self.parser.set_language(Language('util/py-tree-sitter-languages.so', language))

    def process_dee(self, nwo, ext) -> List[Dict[str, Any]]:
        # Process dependees (libraries) to get function implementations
        indexes = []
        _, nwo = remap_nwo(nwo)
        if nwo is None:
            return indexes

        tmp_dir = download(nwo)
        files = walk(tmp_dir, ext)
        # files = glob.iglob(tmp_dir.name + '/**/*.{}'.format(ext), recursive=True)
        sha = None

        for f in files:
            definitions = self.get_function_definitions(f)
            if definitions is None:
                continue
            if sha is None:
                sha = get_sha(tmp_dir, nwo)

            nwo, path, functions = definitions
            indexes.extend(
                (
                    self.extract_function_data(func, nwo, path, sha)
                    for func in functions
                    if len(func["function_tokens"]) > 1
                )
            )
        return indexes

    def process_dee_local(self, nwo, ext) -> List[Dict[str, Any]]:
        # Process dependees (libraries) to get function implementations
        indexes = []
        tmp_dir = download_local(nwo)
        files = walk(tmp_dir, ext)
        sha = None

        for f in files:
            definitions = self.get_function_definitions(f)
            if definitions is None:
                continue
            if sha is None:
                sha = get_sha(tmp_dir, nwo)

            nwo, path, functions = definitions
            indexes.extend(
                (
                    self.extract_function_data(func, nwo, path, sha)
                    for func in functions
                    if len(func["function_tokens"]) > 1
                )
            )
        return indexes

    def process_dee_local_for_DefextsKotlin(
        self, nwo, ext, extract_type
    ) -> List[Dict[str, Any]]:
        # Process dependees (libraries) to get function implementations
        indexes = []
        tmp_dir = download_local_for_DefextsKotlin(nwo)
        files = walk(tmp_dir, "kotlin")
        sha = None

        for f in files:
            definitions = self.get_function_definitions(f)
            if definitions is None:
                continue
            if sha is None:
                sha = get_sha(
                    tmp_dir, "-".join(nwo.split("-")[:-1]).replace("\ufeff", "")
                )

            nwo, path, functions = definitions
            indexes.extend(
                (
                    self.extract_function_data(func, nwo, path, sha)
                    for func in functions
                    if len(func["function_tokens"]) > 1
                )
            )
        return indexes

    def process_dee_local_with_SHA(
        self,
        nwo,
        ext,
        SHA,
        bug2SHA,
        Identifier2Project,
        bug_location_path,
        extract_type,
    ) -> List[Dict[str, Any]]:
        # Process dependees (libraries) to get function implementations
        indexes = []
        if nwo == "jfreechart":
            tmp_dir = download_local_svn_with_SHA(nwo, SHA)
        else:
            tmp_dir = download_local(nwo)
            tmp_dir = go_to_sha(tmp_dir, nwo, SHA)
        files = walk(tmp_dir, ext)
        sha = None

        bug_id = list(bug2SHA.keys())[list(bug2SHA.values()).index(SHA)]
        project_identifier = list(Identifier2Project.keys())[
            list(Identifier2Project.values()).index(nwo)
        ]
        buggy_lines_location = open(
            os.path.join(
                bug_location_path, project_identifier + "-" + bug_id + ".buggy.lines"
            )
        )
        buggy_lines = []
        for line in buggy_lines_location:
            if "FAULT_OF_OMISSION" in line:
                continue
            else:
                buggy_line = {
                    "file": line.split("#")[0],
                    "position": line.split("#")[1],
                    "content": line.split("#")[2],
                }
                buggy_lines.append(buggy_line)

        for f in files:
            if "function" in extract_type:
                definitions = self.get_function_definitions(f)
            elif "block" in extract_type:
                definitions = self.get_code_blocks(f)
            if definitions is None:
                continue
            if sha is None:
                if nwo == "jfreechart":
                    sha = get_sha_svn(tmp_dir, nwo)
                else:
                    sha = get_sha(tmp_dir, nwo)

            nwo, path, functions = definitions

            if "with_bug" not in extract_type:
                buggy_lines_in_f = []
                buggy_part_in_f = []
                for i in range(len(buggy_lines)):
                    if buggy_lines[i]["file"] in f:
                        buggy_lines_in_f.append(buggy_lines[i])

                for func in functions:
                    buggy_lines_in_func = []
                    for line in buggy_lines_in_f:
                        if int(line["position"]) >= int(
                            func["start_point"][0] + 1
                        ) and int(line["position"]) <= int(func["end_point"][0] + 1):
                            buggy_lines_in_func.append(line)

                    if "without_bug" in extract_type:
                        if len(buggy_lines_in_func) > 0:
                            buggy_part_in_f.append(functions.index(func))
                        continue

                    function_token_ids_for_removal = []
                    idx_min = 0
                    for line in buggy_lines_in_func:
                        (
                            function_tokens_position_in_function,
                            _,
                        ) = tokenizations.get_alignments(
                            func["function_tokens"], func["function"].strip()
                        )
                        buggy_line_start_idx = (
                            func["function"].strip()[idx_min:].find(line["content"])
                        )
                        buggy_line_start_idx = buggy_line_start_idx + idx_min
                        buggy_line_idx_list = [
                            buggy_line_start_idx + j
                            for j in range(len(line["content"]))
                        ]
                        idx_min = buggy_line_idx_list[-1]
                        for s in range(len(function_tokens_position_in_function)):
                            if not set(
                                function_tokens_position_in_function[s]
                            ).isdisjoint(buggy_line_idx_list):
                                function_token_ids_for_removal.append(s)
                        print("buggy line: ")
                        print(line)
                        print("Delete tokens: ")
                        print(
                            [
                                func["function_tokens"][k]
                                for k in function_token_ids_for_removal
                            ]
                        )
                        print("Delete idx:", buggy_line_idx_list)
                        print("*" * 20)
                    delete_multiple_element(
                        func["function_tokens"], function_token_ids_for_removal
                    )

                if "without_bug" in extract_type:
                    delete_multiple_element(functions, buggy_part_in_f)

            indexes.extend(
                (
                    self.extract_function_data(func, nwo, path, sha)
                    for func in functions
                    if len(func["function_tokens"]) > 1
                )
            )
        return indexes

    def process_dee_local_oldest(
        self,
        nwo,
        ext,
        sha_list,
        bug2SHA,
        Identifier2Project,
        bug_location_path,
        extract_type,
    ) -> List[Dict[str, Any]]:
        # Process dependees (libraries) to get function implementations
        indexes = []
        if nwo == "jfreechart":
            oldest_sha = get_oldest_sha_svn(nwo, sha_list)
            tmp_dir = download_local_svn_with_SHA(nwo, oldest_sha)
        else:
            tmp_dir = download_local(nwo)
            oldest_sha = get_oldest_sha(tmp_dir, nwo, sha_list)
            tmp_dir = go_to_sha(tmp_dir, nwo, oldest_sha)
        files = walk(tmp_dir, ext)
        sha = None

        bug_id = list(bug2SHA.keys())[list(bug2SHA.values()).index(oldest_sha)]
        project_identifier = list(Identifier2Project.keys())[
            list(Identifier2Project.values()).index(nwo)
        ]
        buggy_lines_location = open(
            os.path.join(
                bug_location_path, project_identifier + "-" + bug_id + ".buggy.lines"
            )
        )
        buggy_lines = []
        for line in buggy_lines_location:
            if "FAULT_OF_OMISSION" in line:
                continue
            else:
                buggy_line = {
                    "file": line.split("#")[0],
                    "position": line.split("#")[1],
                    "content": line.split("#")[2],
                }
                buggy_lines.append(buggy_line)

        for f in files:
            if "function" in extract_type:
                definitions = self.get_function_definitions(f)
            elif "block" in extract_type:
                definitions = self.get_code_blocks(f)
            if definitions is None:
                continue
            if sha is None:
                if nwo == "jfreechart":
                    sha = get_sha_svn(tmp_dir, nwo)
                else:
                    sha = get_sha(tmp_dir, nwo)

            nwo, path, functions = definitions

            if "with_bug" not in extract_type:
                buggy_lines_in_f = []
                buggy_part_in_f = []
                for i in range(len(buggy_lines)):
                    if buggy_lines[i]["file"] in f:
                        buggy_lines_in_f.append(buggy_lines[i])

                for func in functions:
                    buggy_lines_in_func = []
                    for line in buggy_lines_in_f:
                        if int(line["position"]) >= int(
                            func["start_point"][0] + 1
                        ) and int(line["position"]) <= int(func["end_point"][0] + 1):
                            buggy_lines_in_func.append(line)

                    if "without_bug" in extract_type:
                        if len(buggy_lines_in_func) > 0:
                            buggy_part_in_f.append(functions.index(func))
                        continue

                    function_token_ids_for_removal = []
                    idx_min = 0
                    for line in buggy_lines_in_func:
                        (
                            function_tokens_position_in_function,
                            _,
                        ) = tokenizations.get_alignments(
                            func["function_tokens"], func["function"].strip()
                        )
                        buggy_line_start_idx = (
                            func["function"].strip()[idx_min:].find(line["content"])
                        )
                        buggy_line_start_idx = buggy_line_start_idx + idx_min
                        buggy_line_idx_list = [
                            buggy_line_start_idx + j
                            for j in range(len(line["content"]))
                        ]
                        idx_min = buggy_line_idx_list[-1]
                        for s in range(len(function_tokens_position_in_function)):
                            if not set(
                                function_tokens_position_in_function[s]
                            ).isdisjoint(buggy_line_idx_list):
                                function_token_ids_for_removal.append(s)
                        print("buggy line: ")
                        print(line)
                        print("Delete tokens: ")
                        print(
                            [
                                func["function_tokens"][k]
                                for k in function_token_ids_for_removal
                            ]
                        )
                        print("Delete idx:", buggy_line_idx_list)
                        print("*" * 20)
                    delete_multiple_element(
                        func["function_tokens"], function_token_ids_for_removal
                    )

                if "without_bug" in extract_type:
                    delete_multiple_element(functions, buggy_part_in_f)

            indexes.extend(
                (
                    self.extract_function_data(func, nwo, path, sha)
                    for func in functions
                    if len(func["function_tokens"]) > 1
                )
            )
        return indexes, oldest_sha

    def process_dee_local_oldest_by_block(
        self, nwo, ext, sha_list, bug2SHA, Identifier2Project, bug_location_path
    ) -> List[Dict[str, Any]]:
        # Process dependees (libraries) to get code blocks
        indexes = []
        if nwo == "jfreechart":
            oldest_sha = get_oldest_sha_svn(nwo, sha_list)
            tmp_dir = download_local_svn_with_SHA(nwo, oldest_sha)
        else:
            tmp_dir = download_local(nwo)
            oldest_sha = get_oldest_sha(tmp_dir, nwo, sha_list)
            tmp_dir = go_to_sha(tmp_dir, nwo, oldest_sha)
        files = walk(tmp_dir, ext)
        sha = None

        bug_id = list(bug2SHA.keys())[list(bug2SHA.values()).index(oldest_sha)]
        project_identifier = list(Identifier2Project.keys())[
            list(Identifier2Project.values()).index(nwo)
        ]
        buggy_lines_location = open(
            os.path.join(
                bug_location_path, project_identifier + "-" + bug_id + ".buggy.lines"
            )
        )
        buggy_lines = []
        for line in buggy_lines_location:
            if "FAULT_OF_OMISSION" in line:
                continue
            else:
                buggy_line = {
                    "file": line.split("#")[0],
                    "position": line.split("#")[1],
                    "content": line.split("#")[2],
                }
                buggy_lines.append(buggy_line)

        for f in files:
            definitions = self.get_code_blocks(f)
            if definitions is None:
                continue
            if sha is None:
                if nwo == "jfreechart":
                    sha = get_sha_svn(tmp_dir, nwo)
                else:
                    sha = get_sha(tmp_dir, nwo)

            nwo, path, functions = definitions

            buggy_lines_in_f = []
            for i in range(len(buggy_lines)):
                if buggy_lines[i]["file"] in f:
                    buggy_lines_in_f.append(buggy_lines[i])

            for func in functions:
                buggy_lines_in_func = []
                for line in buggy_lines_in_f:
                    if int(line["position"]) >= int(func["start_point"][0] + 1) and int(
                        line["position"]
                    ) <= int(func["end_point"][0] + 1):
                        buggy_lines_in_func.append(line)

                function_token_ids_for_removal = []
                idx_min = 0
                for line in buggy_lines_in_func:
                    (
                        function_tokens_position_in_function,
                        _,
                    ) = tokenizations.get_alignments(
                        func["function_tokens"], func["function"].strip()
                    )
                    buggy_line_start_idx = (
                        func["function"].strip()[idx_min:].find(line["content"])
                    )
                    buggy_line_start_idx = buggy_line_start_idx + idx_min
                    buggy_line_idx_list = [
                        buggy_line_start_idx + j for j in range(len(line["content"]))
                    ]
                    idx_min = buggy_line_idx_list[-1]
                    for s in range(len(function_tokens_position_in_function)):
                        if not set(function_tokens_position_in_function[s]).isdisjoint(
                            buggy_line_idx_list
                        ):
                            function_token_ids_for_removal.append(s)
                    print("buggy line: ")
                    print(line)
                    print("Delete tokens: ")
                    print(
                        [
                            func["function_tokens"][k]
                            for k in function_token_ids_for_removal
                        ]
                    )
                    print("Delete idx:", buggy_line_idx_list)
                    print("*" * 20)
                delete_multiple_element(
                    func["function_tokens"], function_token_ids_for_removal
                )

            indexes.extend(
                (
                    self.extract_function_data(func, nwo, path, sha)
                    for func in functions
                    if len(func["function_tokens"]) > 1
                )
            )
        return indexes, oldest_sha

    def process_dent(
        self, nwo, ext, library_candidates
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
        # Process dependents (applications) to get function calls
        dents = []
        edges = []
        _, nwo = remap_nwo(nwo)
        if nwo is None:
            return dents, edges

        tmp_dir = download(nwo)
        files = walk(tmp_dir, ext)
        sha = None

        for f in files:
            context_and_calls = self.get_context_and_function_calls(f)
            if context_and_calls is None:
                continue
            if sha is None:
                sha = get_sha(tmp_dir, nwo)

            nwo, path, context, calls = context_and_calls
            libraries = []
            for cxt in context:
                if type(cxt) == dict:
                    libraries.extend([v.split(".")[0] for v in cxt.values()])
                elif type(cxt) == list:
                    libraries.extend(cxt)

            match_scopes = {}
            for cxt in set(libraries):
                if cxt in library_candidates:
                    match_scopes[cxt] = library_candidates[cxt]

            for call in calls:
                for (
                    depended_library_name,
                    dependend_library_functions,
                ) in match_scopes.items():
                    for depended_library_function in dependend_library_functions:
                        # Other potential filters: len(call['identifier']) > 6 or len(call['identifier'].split('_')) > 1
                        if call[
                            "identifier"
                        ] not in self.language_parser.STOPWORDS and (
                            (
                                depended_library_function["identifier"].split(".")[-1]
                                == "__init__"
                                and call["identifier"]
                                == depended_library_function["identifier"].split(".")[0]
                            )
                            or (
                                (
                                    len(call["identifier"]) > 9
                                    or (
                                        not call["identifier"].startswith("_")
                                        and len(call["identifier"].split("_")) > 1
                                    )
                                )
                                and call["identifier"]
                                == depended_library_function["identifier"]
                            )
                        ):
                            dent = {
                                "nwo": nwo,
                                "sha": sha,
                                "path": path,
                                "language": self.language,
                                "identifier": call["identifier"],
                                "argument_list": call["argument_list"],
                                "url": "https://github.com/{}/blob/{}/{}#L{}-L{}".format(
                                    nwo,
                                    sha,
                                    path,
                                    call["start_point"][0] + 1,
                                    call["end_point"][0] + 1,
                                ),
                            }
                            dents.append(dent)
                            edges.append(
                                (dent["url"], depended_library_function["url"])
                            )
        return dents, edges

    def process_single_file(self, filepath: PathLike) -> List[Dict[str, Any]]:
        definitions = self.get_function_definitions(filepath)
        if definitions is None:
            return []
        _, _, functions = definitions

        return [
            self.extract_function_data(func, "", "", "")
            for func in functions
            if len(func["function_tokens"]) > 1
        ]

    def extract_function_data(self, function: Dict[str, Any], nwo, path: str, sha: str):
        return {
            "nwo": nwo,
            "sha": sha,
            "path": path,
            "language": self.language,
            "identifier": function["identifier"],
            "parameters": function.get("parameters", ""),
            "argument_list": function.get("argument_list", ""),
            "return_statement": function.get("return_statement", ""),
            "docstring": function["docstring"].strip(),
            "docstring_summary": function["docstring_summary"].strip(),
            "docstring_tokens": tokenize_docstring(function["docstring_summary"]),
            "function": function["function"].strip(),
            "function_tokens": function["function_tokens"],
            "url": "https://github.com/{}/blob/{}/{}#L{}-L{}".format(
                nwo,
                sha,
                path,
                function["start_point"][0] + 1,
                function["end_point"][0] + 1,
            ),
        }

    def get_context_and_function_calls(
        self, filepath: str
    ) -> Optional[Tuple[str, str, List, List]]:
        nwo = "/".join(filepath.split("/")[3:5])
        path = "/".join(filepath.split("/")[5:])
        if any(fp in path.lower() for fp in self.language_parser.FILTER_PATHS):
            return None
        try:
            with open(filepath) as source_code:
                blob = source_code.read()
            tree = DataProcessor.PARSER.parse(blob.encode())
            return (
                nwo,
                path,
                self.language_parser.get_context(tree, blob),
                self.language_parser.get_calls(tree, blob),
            )
        except (
            UnicodeDecodeError,
            FileNotFoundError,
            IsADirectoryError,
            ValueError,
            OSError,
        ):
            return None

    def get_function_definitions(
        self, filepath: str
    ) -> Optional[Tuple[str, str, List]]:
        nwo = "/".join(filepath.split("/")[3:5])
        path = "/".join(filepath.split("/")[5:])
        if any(fp in path.lower() for fp in self.language_parser.FILTER_PATHS):
            return None
        try:
            with open(filepath) as source_code:
                blob = source_code.read()
            tree = DataProcessor.PARSER.parse(blob.encode())
            # print(tree)
            return (nwo, path, self.language_parser.get_definition(tree, blob))
        except (
            UnicodeDecodeError,
            FileNotFoundError,
            IsADirectoryError,
            ValueError,
            OSError,
        ):
            return None

    def get_code_blocks(self, filepath: str) -> Optional[Tuple[str, str, List]]:
        nwo = "/".join(filepath.split("/")[3:5])
        path = "/".join(filepath.split("/")[5:])
        if any(fp in path.lower() for fp in self.language_parser.FILTER_PATHS):
            return None
        try:
            with open(filepath) as source_code:
                blob = source_code.read()
            tree = DataProcessor.PARSER.parse(blob.encode())
            with open(filepath) as source_code_lines:
                blob_lines = source_code_lines.readlines()
            # tokens = []
            # tokens_type = []
            # self.tree_v.create_node(tree.root_node.type, 0)
            # self.dfs(blob.encode(), tree.root_node, tokens, tokens_type, 1, 0, "type")
            # self.tree_v.save2file("/home/Project/dl/repair_plus/tree_type.txt")
            # print(self.tree_v)
            return (nwo, path, self.language_parser.get_block(tree, blob, blob_lines))
        except (
            UnicodeDecodeError,
            FileNotFoundError,
            IsADirectoryError,
            ValueError,
            OSError,
        ):
            return None

    def dfs(self, code, node, tokens, tokens_type, depth, id, style):
        if len(node.children) == 0:
            snippet = code[node.start_byte : node.end_byte].strip(b" ")
            if isinstance(snippet, bytes):
                snippet = snippet.decode("utf8")
            if len(snippet) > 0:
                tokens.append(snippet)
                tokens_type.append(node.type)
            # self.tree.create_node(snippet, id*100 + 1, parent = id)
            return
        idx = 0
        for child in node.children:
            idx += 1
            snippet = code[child.start_byte : child.end_byte].strip(b" ")
            if isinstance(snippet, bytes):
                snippet = snippet.decode("utf8")
            if style == "type":
                self.tree_v.create_node(child.type, id * 100 + idx, parent=id)
            elif style == "code":
                self.tree_v.create_node(snippet, id * 100 + idx, parent=id)
            self.dfs(code, child, tokens, tokens_type, depth + 1, id * 100 + idx, style)


if __name__ == "__main__":
    args = docopt(__doc__)

    repository_dependencies = pd.read_csv(
        args["INPUT_DIR"] + "repository_dependencies-1.4.0-2018-12-22.csv",
        index_col=False,
    )
    projects = pd.read_csv(
        args["INPUT_DIR"] + "projects_with_repository_fields-1.4.0-2018-12-22.csv",
        index_col=False,
    )

    repository_dependencies["Manifest Platform"] = repository_dependencies[
        "Manifest Platform"
    ].apply(lambda x: x.lower())
    id_to_nwo = {
        project["ID"]: project["Repository Name with Owner"]
        for project in projects[["ID", "Repository Name with Owner"]]
        .dropna()
        .to_dict(orient="records")
    }
    nwo_to_name = {
        project["Repository Name with Owner"]: project["Name"]
        for project in projects[["Repository Name with Owner", "Name"]]
        .dropna()
        .to_dict(orient="records")
    }

    filtered = (
        repository_dependencies[
            (repository_dependencies["Host Type"] == "GitHub")
            & (
                repository_dependencies["Manifest Platform"]
                == LANGUAGE_METADATA[args["--language"]]["platform"]
            )
        ][["Repository Name with Owner", "Dependency Project ID"]]
        .dropna()
        .to_dict(orient="records")
    )

    dependency_pairs = [
        (rd["Repository Name with Owner"], id_to_nwo[int(rd["Dependency Project ID"])])
        for rd in filtered
        if int(rd["Dependency Project ID"]) in id_to_nwo
    ]

    dependency_pairs = list(set(dependency_pairs))

    dents, dees = zip(*dependency_pairs)
    # dents = list(set(dents))
    dees = list(set(dees))

    if args["--language"] == "kotlin":
        args[
            "--tree-sitter-build"
        ] = "/home/Project/dl/repair_plus/finetune/util/py-tree-sitter-kotlin.so"
        DataProcessor.PARSER.set_language(
            Language(args["--tree-sitter-build"], args["--language"])
        )
    else:
        DataProcessor.PARSER.set_language(
            Language(args["--tree-sitter-build"], args["--language"])
        )

    processor = DataProcessor(
        language=args["--language"],
        language_parser=LANGUAGE_METADATA[args["--language"]]["language_parser"],
    )

    with Pool(processes=int(args["--processes"])) as pool:
        output = pool.imap_unordered(
            functools.partial(
                processor.process_dee, ext=LANGUAGE_METADATA[args["--language"]]["ext"]
            ),
            dees,
        )

    definitions = list(flatten(output))
    with open(
        args["OUTPUT_DIR"] + "{}_definitions.pkl".format(args["--language"]), "wb"
    ) as f:
        pickle.dump(definitions, f)

    license_filter_file = args.get("--license-filter")
    if license_filter_file is not None:
        with open(license_filter_file, "rb") as f:
            license_filter = pickle.load(f)
        valid_nwos = dict([(l[0], l[3]) for l in license_filter])

        # Sort function definitions with repository popularity
        definitions = [
            dict(list(d.items()) + [("score", valid_nwos[d["nwo"]])])
            for d in definitions
            if d["nwo"] in valid_nwos
        ]
        definitions = sorted(definitions, key=lambda x: -x["score"])

        # dedupe
        seen = set()
        filtered = []
        for d in definitions:
            if " ".join(d["function_tokens"]) not in seen:
                filtered.append(d)
                seen.add(" ".join(d["function_tokens"]))

        dd = DuplicateDetector(min_num_tokens_per_document=10)
        filter_mask = [
            dd.add_file(id=idx, tokens=d["function_tokens"], language=d["language"])
            for idx, d in enumerate(filtered)
        ]
        exclusion_set = dd.compute_ids_to_exclude()
        exclusion_mask = [idx not in exclusion_set for idx, _ in enumerate(filtered)]
        filtered = [
            d
            for idx, d in enumerate(filtered)
            if filter_mask[idx] & exclusion_mask[idx]
        ]

        with open(
            args["OUTPUT_DIR"] + "{}_dedupe_definitions.pkl".format(args["--language"]),
            "wb",
        ) as f:
            pickle.dump(filtered, f)
