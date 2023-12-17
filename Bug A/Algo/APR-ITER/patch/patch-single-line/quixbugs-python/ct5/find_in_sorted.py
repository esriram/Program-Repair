--- 
+++ 
@@ -6,7 +6,7 @@
         if x < arr[mid]:
             return binsearch(start, mid)
         elif x > arr[mid]:
-            return binsearch(mid, end)
+            return binsearch(mid+1, end)# 实例化应该上一个词变成数组,==( x,)
         else:
             return mid
 
