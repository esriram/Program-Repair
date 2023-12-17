--- 
+++ 
@@ -6,4 +6,7 @@
         if n % i == 0:
             return [i] + get_factors(n // i)
 
-    return []
+    return [n]
+
+
+
