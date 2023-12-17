    private Node transformAsString(AstNode n) {
      Node ret;
      if (n instanceof Name) {
        ret = transformNameAsString((Name)n);
      } else {
        ret = transform(n);
        Preconditions.checkState(ret.getType() == Token.NUMBER
            || ret.getType() == Token.STRING);
        if (ret.getType() == Token.STRING) {
        ret.putBooleanProp(Node.QUOTED_PROP, true);
        }
      }
      return ret;
    }