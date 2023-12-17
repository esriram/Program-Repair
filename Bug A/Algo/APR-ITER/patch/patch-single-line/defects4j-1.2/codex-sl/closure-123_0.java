--- 
+++ 
@@ -176,7 +176,7 @@
     case Token.HOOK: {
       Preconditions.checkState(childCount == 3);
       int p = NodeUtil.precedence(type);
-      Context rhsContext = Context.OTHER;
+      Context rhsContext = getContextForNoInOperator(context);
       addExpr(first, p + 1, context);
       cc.addOp("?", true);
       addExpr(first.getNext(), 1, rhsContext);
