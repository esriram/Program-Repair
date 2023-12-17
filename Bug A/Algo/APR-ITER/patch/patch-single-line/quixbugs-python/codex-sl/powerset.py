--- 
+++ 
@@ -2,6 +2,6 @@
     if arr:
         first, *rest = arr
         rest_subsets = powerset(rest)
-        return [[first] + subset for subset in rest_subsets]
+        return rest_subsets + [[first] + subset for subset in rest_subsets]
     else:
         return [[]]
