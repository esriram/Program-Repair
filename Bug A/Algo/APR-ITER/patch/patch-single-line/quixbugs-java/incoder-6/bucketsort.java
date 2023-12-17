--- 
+++ 
@@ -6,7 +6,7 @@
 
     ArrayList<Integer> sorted_arr = new ArrayList<Integer>(100);
     int i = 0;
-    for (Integer count : arr) { // arr is counts in fixed version
+    for (int count : counts) {
     sorted_arr.addAll(Collections.nCopies(count, i));
     i++;
     }
