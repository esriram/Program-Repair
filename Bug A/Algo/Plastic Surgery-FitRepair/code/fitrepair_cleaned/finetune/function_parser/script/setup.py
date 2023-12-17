import glob

from tree_sitter import Language

languages = [
    "vendor/tree-sitter-python",
    "vendor/tree-sitter-javascript",
    # '/src/vendor/tree-sitter-typescript/typescript',
    # '/src/vendor/tree-sitter-typescript/tsx',
    "vendor/tree-sitter-go",
    "vendor/tree-sitter-ruby",
    "vendor/tree-sitter-java",
    "vendor/tree-sitter-cpp",
    "vendor/tree-sitter-php",
]

Language.build_library(
    # Store the library in the directory
    "util/py-tree-sitter-languages.so",
    # Include one or more languages
    languages,
)
