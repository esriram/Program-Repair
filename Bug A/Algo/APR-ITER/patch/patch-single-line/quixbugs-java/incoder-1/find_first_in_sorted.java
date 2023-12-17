--- 
+++ 
@@ -2,7 +2,7 @@
     int lo = 0;
     int hi = arr.length;
 
-    while (lo <= hi) {
+    while (lo < hi) {
         int mid = (lo + hi) / 2; // check if this is floor division
 
         if (x == arr[mid] && (mid == 0 || x != arr[mid-1])) {
