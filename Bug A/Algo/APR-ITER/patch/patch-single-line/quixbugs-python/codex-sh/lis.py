--- 
+++ 
@@ -10,6 +10,6 @@
 
         if length == longest or val < arr[ends[length + 1]]:
             ends[length + 1] = i
-            longest = length + 1
+            longest = max(longest, length + 1)
 
     return longest
