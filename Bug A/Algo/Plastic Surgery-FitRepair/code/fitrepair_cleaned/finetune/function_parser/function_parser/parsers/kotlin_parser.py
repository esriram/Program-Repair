from typing import Any, Dict, List

from parsers.commentutils import get_docstring_summary, strip_c_style_comment_delimiters
from parsers.language_parser import (
    LanguageParser,
    match_from_span,
    tokenize_code,
    traverse_type,
)


class KotlinParser(LanguageParser):

    # FILTER_PATHS = ('test', 'tests')
    FILTER_PATHS = ()

    # BLACKLISTED_FUNCTION_NAMES = {'toString', 'hashCode', 'equals', 'finalize', 'notify', 'notifyAll', 'clone'}
    BLACKLISTED_FUNCTION_NAMES = {}

    @staticmethod
    def get_definition(tree, blob: str) -> List[Dict[str, Any]]:
        classes = (
            node for node in tree.root_node.children if node.type == "class_declaration"
        )

        definitions = []
        for _class in classes:
            class_identifier = match_from_span(
                [child for child in _class.children if "identifier" in child.type][0],
                blob,
            ).strip()
            for child in (
                child for child in _class.children if child.type == "class_body"
            ):
                for idx, node in enumerate(child.children):
                    if node.type == "function_declaration":
                        if KotlinParser.is_function_body_empty(node):
                            continue
                        docstring = ""
                        # if idx - 1 >= 0:
                        #     print(child.children[idx-1].type)
                        if idx - 1 >= 0 and "comment" in child.children[idx - 1].type:
                            docstring = match_from_span(child.children[idx - 1], blob)
                            docstring = strip_c_style_comment_delimiters(docstring)
                        docstring_summary = get_docstring_summary(docstring)

                        metadata = KotlinParser.get_function_metadata(node, blob)
                        if (
                            metadata["identifier"]
                            in KotlinParser.BLACKLISTED_FUNCTION_NAMES
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

        for idx, node in enumerate(tree.root_node.children):
            if node.type == "function_declaration":
                if KotlinParser.is_function_body_empty(node):
                    continue
                docstring = ""
                # if idx - 1 >= 0:
                #     print(child.children[idx-1].type)
                if idx - 1 >= 0 and "comment" in tree.root_node.children[idx - 1].type:
                    docstring = match_from_span(tree.root_node.children[idx - 1], blob)
                    docstring = strip_c_style_comment_delimiters(docstring)
                docstring_summary = get_docstring_summary(docstring)

                metadata = KotlinParser.get_function_metadata(node, blob)
                if metadata["identifier"] in KotlinParser.BLACKLISTED_FUNCTION_NAMES:
                    continue
                definitions.append(
                    {
                        "type": node.type,
                        "identifier": "",
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
                if "identifier" in n.type:
                    metadata["identifier"] = match_from_span(n, blob).strip("(:")
                elif n.type == "value_arguments":
                    metadata["argument_list"] = match_from_span(n, blob)
            if n.type == "class":
                is_header = True
            elif n.type == ":":
                break
        return metadata

    @staticmethod
    def is_function_body_empty(node):
        for c in node.children:
            if c.type in {"function_body"}:
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
            if "identifier" in n.type:
                metadata["identifier"] = match_from_span(n, blob).strip("(")
            elif n.type == "parameter":
                parameters.append(match_from_span(n, blob))
        metadata["parameters"] = " ".join(parameters)
        return metadata
