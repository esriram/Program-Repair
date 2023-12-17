--- 
+++ 
@@ -4,7 +4,7 @@
 
     nodesvisited.add(startnode);
 
-    while (true) {
+    while (!queue.isEmpty()) {
         Node node = queue.removeFirst();
 
         if (node == goalnode) {
