--- 
+++ 
@@ -25,8 +25,7 @@
                 unvisitedNodes.put(nextnode, Integer.MAX_VALUE);
             }
 
-            unvisitedNodes.put(nextnode, Math.min(unvisitedNodes.get(nextnode),
-                    unvisitedNodes.get(nextnode) + length_by_edge.get(Arrays.asList(node, nextnode))));
+            unvisitedNodes.put(nextnode, Math.min(unvisitedNodes.get(nextnode), distance + length_by_edge.get(Arrays.asList(node, nextnode))));
         }
     }
 
