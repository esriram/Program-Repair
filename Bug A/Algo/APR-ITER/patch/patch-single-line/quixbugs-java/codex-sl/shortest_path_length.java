--- 
+++ 
@@ -26,7 +26,7 @@
             }
 
             unvisitedNodes.put(nextnode, Math.min(unvisitedNodes.get(nextnode),
-                    unvisitedNodes.get(nextnode) + length_by_edge.get(Arrays.asList(node, nextnode))));
+                    distance + length_by_edge.get(Arrays.asList(node, nextnode))));
         }
     }
 
