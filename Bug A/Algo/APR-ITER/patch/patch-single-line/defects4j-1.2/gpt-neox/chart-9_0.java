--- 
+++ 
@@ -24,7 +24,7 @@
         endIndex = -(endIndex + 1); // this is first item AFTER end period
         endIndex = endIndex - 1;    // so this is last item BEFORE end
     }
-    if (endIndex < 0) {
+    if (startIndex < 0 || endIndex < 0 || startIndex > endIndex) {
         emptyRange = true;
     }
     if (emptyRange) {
