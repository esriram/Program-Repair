--- 
+++ 
@@ -6,4 +6,8 @@
         if n % i == 0:
             return [i] + get_factors(n // i)
 
-    return []
+    return [n]
+
+
+
+
