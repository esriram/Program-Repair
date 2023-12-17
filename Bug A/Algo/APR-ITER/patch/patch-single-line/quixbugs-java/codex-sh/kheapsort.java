--- 
+++ 
@@ -5,7 +5,8 @@
     }
 
     ArrayList<Integer> output = new ArrayList<Integer>();
-    for (Integer x : arr) {
+
+    for (Integer x : arr.subList(k, arr.size())) {
         heap.add(x);
         Integer popped = heap.poll();
         output.add(popped);
