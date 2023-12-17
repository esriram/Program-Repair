--- 
+++ 
@@ -9,6 +9,6 @@
     if k < num_less:
         return kth(below, k)
     elif k >= num_lessoreq:
-        return kth(above, k)
+        return kth(above, k - num_lessoreq)
     else:
         return pivot
