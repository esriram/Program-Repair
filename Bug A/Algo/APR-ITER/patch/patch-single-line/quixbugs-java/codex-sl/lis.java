--- 
+++ 
@@ -16,7 +16,7 @@
 
         if (length == longest || val < arr[ends.get(length+1)]) {
             ends.put(length+1, i);
-            longest = length + 1;
+            longest = Math.max(longest, length+1);
         }
 
         i++;
