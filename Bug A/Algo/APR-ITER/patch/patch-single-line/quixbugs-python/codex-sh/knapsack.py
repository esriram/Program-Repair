--- 
+++ 
@@ -7,7 +7,7 @@
         for j in range(1, capacity + 1):
             memo[i, j] = memo[i - 1, j]
 
-            if  weight < j:
+            if weight <= j:
                 memo[i, j] = max(
                     memo[i, j],
                     value + memo[i - 1, j - weight]
