--- 
+++ 
@@ -8,7 +8,7 @@
             for j in range(n):
                 length_by_path[i, j] = min(
                     length_by_path[i, j],
-                    length_by_path[i, k] + length_by_path[j, k]
+                    length_by_path[i, k] + length_by_path[k, j],
                 )
 
     return length_by_path
