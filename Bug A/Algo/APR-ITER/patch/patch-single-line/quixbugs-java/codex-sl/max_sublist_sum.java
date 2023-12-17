--- 
+++ 
@@ -3,7 +3,7 @@
     int max_so_far = 0;
 
     for (int x : arr) {
-        max_ending_here = max_ending_here + x;
+        max_ending_here = Math.max(0, max_ending_here + x);
         max_so_far = Math.max(max_so_far, max_ending_here);
     }
 
