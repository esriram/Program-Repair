--- 
+++ 
@@ -16,7 +16,9 @@
 
         if (length == longest || val < arr[ends.get(length+1)]) {
             ends.put(length+1, i);
-            longest = length + 1;
+            if (length+1 > longest) {
+                longest = length+1;
+            }
         }
 
         i++;
