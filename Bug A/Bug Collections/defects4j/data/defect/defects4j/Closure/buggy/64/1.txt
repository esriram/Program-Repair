  String toSource(Node n) {
    initCompilerOptionsIfTesting();
    return toSource(n, null);
  }