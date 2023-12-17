    private void report(NodeTraversal t, Node n, DiagnosticType error,
        String... arguments) {
      compiler.report(t.makeError(n, error, arguments));
      hasErrors = true;
    }