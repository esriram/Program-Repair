--- 
+++ 
@@ -10,7 +10,7 @@
     for (Integer x : arr.subList(1, arr.size())) {
         if (x < pivot) {
             lesser.add(x);
-        } else if (x > pivot) {
+        } else {
             greater.add(x);
         }
     }
