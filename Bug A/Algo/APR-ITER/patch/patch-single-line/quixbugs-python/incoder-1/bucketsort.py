--- 
+++ 
@@ -4,7 +4,8 @@
         counts[x] += 1
 
     sorted_arr = []
-    for i, count in enumerate(arr):
+    for i in range(k):
+        count = counts[i]
         sorted_arr.extend([i] * count)
 
     return sorted_arr
