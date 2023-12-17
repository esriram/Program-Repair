# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import re
from pathlib import Path

from tree_sitter import Language, Parser
from treelib import Node, Tree


class TreeSitterProcessor:
    def __init__(self, language):
        self.language = language
        self.lib_folder = Path("./util")
        # change to your own path
        self.root_folder = Path("/home/Project/dl/repair_plus/util")
        self.root_folder.is_dir(), f"{self.root_folder} is not a directory."
        self.parser = None

        self.assignment = ["assignment", "augmented_assignment", "for_in_clause"]
        self.if_statement = ["if_statement"]
        self.for_statement = ["for_statement"]
        self.while_statement = ["while_statement"]
        self.do_first_statement = ["for_in_clause"]
        self.def_statement = ["default_parameter"]

        self.tree = Tree()

        self.create_treesiter_parser()

    def create_treesiter_parser(self):
        if self.parser is None:
            lib_path = self.lib_folder.joinpath(f"{self.language}.so")
            repo_path = self.root_folder.joinpath(f"tree-sitter-{self.language}")
            if not lib_path.exists():
                assert repo_path.is_dir(), repo_path
                Language.build_library(
                    # Store the library in the `build` directory
                    str(lib_path),
                    # Include one or more languages
                    [str(repo_path)],
                )
            language = Language(lib_path, self.language)
            self.parser = Parser()
            self.parser.set_language(language)

    def get_identifiers(self, code):
        code = code.replace("\r", "")
        code = bytes(code, "utf8")
        tree = self.get_ast(code)
        tokens = []
        tokens_type = []
        self.dfs_plain(code, tree.root_node, tokens, tokens_type)
        identifiers = [
            tokens[x] for x in range(len(tokens_type)) if tokens_type[x] == "identifier"
        ]
        return list(set(identifiers))

    def get_ast_tree(self, code, path, style):
        code = code.replace("\r", "")
        code = bytes(code, "utf8")
        tree = self.get_ast(code)
        tokens = []
        tokens_type = []
        self.tree.create_node(tree.root_node.type, 0)
        self.dfs(code, tree.root_node, tokens, tokens_type, 1, 0, style)
        self.tree.save2file(path)
        # return tokens, tokens_type

    def get_ast(self, code):
        assert isinstance(code, str) or isinstance(code, bytes)
        if isinstance(code, str):
            code = bytes(code, "utf8")
        tree = self.parser.parse(code)
        # print(tree.root_node.sexp())
        return tree

    def dfs_plain(self, code, node, tokens, tokens_type):
        if len(node.children) == 0:
            snippet = code[node.start_byte : node.end_byte].strip(b" ")
            if isinstance(snippet, bytes):
                snippet = snippet.decode("utf8")
            if len(snippet) > 0:
                tokens.append(snippet)
                tokens_type.append(node.type)
            return
        for child in node.children:
            self.dfs_plain(code, child, tokens, tokens_type)

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
                self.tree.create_node(child.type, id * 100 + idx, parent=id)
            elif style == "code":
                self.tree.create_node(snippet, id * 100 + idx, parent=id)
            self.dfs(code, child, tokens, tokens_type, depth + 1, id * 100 + idx, style)
