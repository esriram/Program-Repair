  private Node transformNameAsString(Name node) {
    JSDocInfo jsDocInfo = handleJsDoc(node);
    Node irNode = transformDispatcher.processName(node, true);
    if (jsDocInfo != null) {
      irNode.setJSDocInfo(jsDocInfo);
    }
    setSourceInfo(irNode, node);
    return irNode;
  }