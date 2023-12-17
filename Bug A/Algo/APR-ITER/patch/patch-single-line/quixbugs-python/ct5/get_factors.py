--- 
+++ 
@@ -6,4 +6,5 @@
         if n % i == 0:
             return [i] + get_factors(n // i)
 
-    return []
+    return[n ]
+
