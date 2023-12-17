from typing import Any, Dict, List

from parsers.commentutils import get_docstring_summary, strip_c_style_comment_delimiters
from parsers.language_parser import (
    LanguageParser,
    match_from_span,
    tokenize_code,
    traverse_type,
)


def get_lines_for_tokens(node, blob, lines_of_tokens: List) -> None:
    if node.type == "string":
        line_start = node.start_point[0]
        line_end = node.end_point[0]
        for i in range(line_start, line_end + 1):
            lines_of_tokens[i].append(match_from_span(node, blob))
        return
    for n in node.children:
        get_lines_for_tokens(n, blob, lines_of_tokens)
    if not node.children:
        line_start = node.start_point[0]
        line_end = node.end_point[0]
        for i in range(line_start, line_end + 1):
            lines_of_tokens[i].append(match_from_span(node, blob))


class JavaParser(LanguageParser):

    # FILTER_PATHS = ('test', 'tests')
    FILTER_PATHS = ()

    # BLACKLISTED_FUNCTION_NAMES = {'toString', 'hashCode', 'equals', 'finalize', 'notify', 'notifyAll', 'clone'}
    BLACKLISTED_FUNCTION_NAMES = {}

    @staticmethod
    def get_block(tree, blob, blob_lines) -> List[Dict[str, Any]]:
        comment_blocks = (
            node for node in tree.root_node.children if "comment" in node.type
        )
        code_blocks = (
            node for node in tree.root_node.children if "comment" not in node.type
        )
        doc_interval = []
        for comment in comment_blocks:
            doc_interval += [
                comment.start_point[0] + i
                for i in range(comment.end_point[0] - comment.start_point[0] + 1)
            ]
        lines_of_tokens = [[] for _ in range(len(blob_lines))]
        for node in code_blocks:
            get_lines_for_tokens(node, blob, lines_of_tokens)
        single_definition_length = 300
        definitions = []
        single_definition_tokens = []
        single_definition_string = ""
        single_definition_start_point = -1
        single_definition_end_point = -1
        current_definition_length = 0
        for line_idx in range(len(blob_lines)):
            if line_idx in doc_interval:
                continue
            if single_definition_start_point == -1:
                single_definition_start_point = line_idx
            single_definition_end_point = line_idx
            single_definition_tokens += lines_of_tokens[line_idx]
            current_definition_length += len(lines_of_tokens[line_idx])
            single_definition_string += blob_lines[line_idx]
            if current_definition_length >= single_definition_length:
                definitions.append(
                    {
                        "type": None,
                        "identifier": None,
                        "parameters": None,
                        "function": single_definition_string,
                        "function_tokens": single_definition_tokens,
                        "docstring": "",
                        "docstring_summary": "",
                        "start_point": [
                            single_definition_start_point,
                            single_definition_start_point,
                        ],
                        "end_point": [
                            single_definition_end_point,
                            single_definition_end_point,
                        ],
                    }
                )
                single_definition_tokens = []
                single_definition_string = ""
                single_definition_start_point = -1
                single_definition_end_point = -1
                current_definition_length = 0
        return definitions

    @staticmethod
    def get_definition(tree, blob: str) -> List[Dict[str, Any]]:
        classes = (
            node for node in tree.root_node.children if node.type == "class_declaration"
        )

        definitions = []
        for _class in classes:
            class_identifier = match_from_span(
                [child for child in _class.children if child.type == "identifier"][0],
                blob,
            ).strip()
            for child in (
                child for child in _class.children if child.type == "class_body"
            ):
                for idx, node in enumerate(child.children):
                    if node.type == "method_declaration":
                        if JavaParser.is_method_body_empty(node):
                            continue
                        docstring = ""
                        # if idx - 1 >= 0:
                        #     print(child.children[idx-1].type)
                        if idx - 1 >= 0 and "comment" in child.children[idx - 1].type:
                            docstring = match_from_span(child.children[idx - 1], blob)
                            docstring = strip_c_style_comment_delimiters(docstring)
                        docstring_summary = get_docstring_summary(docstring)

                        metadata = JavaParser.get_function_metadata(node, blob)
                        if (
                            metadata["identifier"]
                            in JavaParser.BLACKLISTED_FUNCTION_NAMES
                        ):
                            continue
                        definitions.append(
                            {
                                "type": node.type,
                                "identifier": "{}.{}".format(
                                    class_identifier, metadata["identifier"]
                                ),
                                "parameters": metadata["parameters"],
                                "function": match_from_span(node, blob),
                                "function_tokens": tokenize_code(node, blob),
                                "docstring": docstring,
                                "docstring_summary": docstring_summary,
                                "start_point": node.start_point,
                                "end_point": node.end_point,
                            }
                        )
        return definitions

    @staticmethod
    def get_class_metadata(class_node, blob: str) -> Dict[str, str]:
        metadata = {
            "identifier": "",
            "argument_list": "",
        }
        is_header = False
        for n in class_node.children:
            if is_header:
                if n.type == "identifier":
                    metadata["identifier"] = match_from_span(n, blob).strip("(:")
                elif n.type == "argument_list":
                    metadata["argument_list"] = match_from_span(n, blob)
            if n.type == "class":
                is_header = True
            elif n.type == ":":
                break
        return metadata

    @staticmethod
    def is_method_body_empty(node):
        for c in node.children:
            if c.type in {"method_body", "constructor_body"}:
                if c.start_point[0] == c.end_point[0]:
                    return True

    @staticmethod
    def get_function_metadata(function_node, blob: str) -> Dict[str, str]:
        metadata = {
            "identifier": "",
            "parameters": "",
        }

        declarators = []
        traverse_type(
            function_node,
            declarators,
            "{}_declaration".format(function_node.type.split("_")[0]),
        )
        parameters = []
        for n in declarators[0].children:
            if n.type == "identifier":
                metadata["identifier"] = match_from_span(n, blob).strip("(")
            elif n.type == "formal_parameter":
                parameters.append(match_from_span(n, blob))
        metadata["parameters"] = " ".join(parameters)
        return metadata
