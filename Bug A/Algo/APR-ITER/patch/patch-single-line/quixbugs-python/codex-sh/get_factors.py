--- 
+++ 
@@ -6,4 +6,4 @@
         if n % i == 0:
             return [i] + get_factors(n // i)
 
-    return []
+    return [n]
