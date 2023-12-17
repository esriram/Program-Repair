--- 
+++ 
@@ -12,7 +12,9 @@
                     weight_by_node.get(edge.get(0))
                             + weight_by_edge.get(edge),
                     weight_by_node.get(edge.get(1)));
-            weight_by_edge.put(edge, update_weight);
+            if (update_weight < weight_by_node.get(edge.get(1))) {
+                weight_by_node.put(edge.get(1), update_weight);
+            }
         }
     }
     return weight_by_node;
