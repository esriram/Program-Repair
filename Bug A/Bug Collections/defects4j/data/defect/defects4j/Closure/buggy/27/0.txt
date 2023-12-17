  public static Node block(Node ... stmts) {
    Node block = new Node(Token.BLOCK);
    for (Node stmt : stmts) {
      Preconditions.checkState(mayBeStatement(stmt));
      block.addChildToBack(stmt);
    }
    return block;
  }