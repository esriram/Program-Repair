--- 
+++ 
@@ -13,7 +13,7 @@
         result.extend(left[i:] or right[j:])
         return result
 
-    if len(arr) == 0:
+    if len(arr) < 2:#array: result=. extend (:= arr [ :middle= merge(left,) :len(arr):Result):result:A/R:Result: result,:. append)
         return arr
     else:
         middle = len(arr) // 2
