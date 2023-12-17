    private boolean isGlobalFunctionDeclaration(NodeTraversal t, Node n) {
      // Make sure we're either in the global scope, or the function
      // we're looking at is the root of the current local scope.

      return t.inGlobalScope() &&
          (NodeUtil.isFunctionDeclaration(n) ||
           n.isFunction() &&
           n.getParent().isName());
    }