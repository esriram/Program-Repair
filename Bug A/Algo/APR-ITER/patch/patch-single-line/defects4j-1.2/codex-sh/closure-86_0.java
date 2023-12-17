--- 
+++ 
@@ -38,8 +38,14 @@
     case Token.NEW:
       // TODO(nicksantos): This needs to be changed so that it
       // returns true iff we're sure the value was never aliased from inside
-      // the constructor (similar to callHasLocalResult)
-      return true;
+      // the constructor (similar to callHasLocalResult).
+      return false;
+    case Token.NULL:
+    case Token.VOID:
+    case Token.TRUE:
+    case Token.FALSE:
+    case Token.NUMBER:
+    case Token.STRING:
     case Token.FUNCTION:
     case Token.REGEXP:
     case Token.ARRAYLIT:
