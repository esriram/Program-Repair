--- 
+++ 
@@ -7,6 +7,7 @@
             } else if (node == goalnode) {
                 return true;
             } else {
+                nodesvisited.add(node);
                 for (Node successornodes : node.getSuccessors()) {
                     if (search(successornodes)) { return true; }
                 }
