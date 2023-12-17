--- 
+++ 
@@ -7,5 +7,4 @@
             depth -= 1
             if depth < 0:
                 return False
-
-    return True
+    return depth == 0
