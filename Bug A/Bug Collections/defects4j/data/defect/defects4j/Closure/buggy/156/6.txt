  private void updateFunctionDeclarationAtFunctionNode(Name n) {

    Ref ref = n.declaration;
    String fnName = ref.node.getString();
    addStubsForUndeclaredProperties(
        n, fnName, ref.node.getAncestor(2), ref.node.getParent());
  }