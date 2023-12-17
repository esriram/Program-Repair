--- 
+++ 
@@ -16,7 +16,7 @@
             a = stack.pop()
             b = stack.pop()
             stack.append(
-                op(token, a, b)
+                op(token, b, a)
             )
 
     return stack.pop()
