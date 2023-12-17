--- 
+++ 
@@ -4,6 +4,7 @@
 
     for (int x : arr) {
         max_ending_here = max_ending_here + x;
+        if (max_ending_here < 0) max_ending_here = 0;
         max_so_far = Math.max(max_so_far, max_ending_here);
     }
 
