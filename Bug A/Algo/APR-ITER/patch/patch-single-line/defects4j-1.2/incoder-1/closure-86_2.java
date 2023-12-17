--- 
+++ 
@@ -39,7 +39,8 @@
       // TODO(nicksantos): This needs to be changed so that it
       // returns true iff we're sure the value was never aliased from inside
       // the constructor (similar to callHasLocalResult)
-      return true;
+      // If it's not, the result will always be false.
+      return false;
     case Token.FUNCTION:
     case Token.REGEXP:
     case Token.ARRAYLIT:
