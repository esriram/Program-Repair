--- 
+++ 
@@ -13,7 +13,7 @@
         result.extend(left[i:] or right[j:])
         return result
 
-    if len(arr) == 0:
+    if len(arr) <= 1:
         return arr
     else:
         middle = len(arr) // 2
