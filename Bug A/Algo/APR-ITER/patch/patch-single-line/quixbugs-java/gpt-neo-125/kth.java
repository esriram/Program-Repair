--- 
+++ 
@@ -16,7 +16,7 @@
     if (k < num_less) {
         return kth(below, k);
     } else if (k >= num_lessoreq) {
-        return kth(above, k);
+        return kth(above, k - num_lessoreq);
     } else {
         return pivot;
     }
