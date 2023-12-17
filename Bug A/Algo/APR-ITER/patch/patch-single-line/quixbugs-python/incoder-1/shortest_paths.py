--- 
+++ 
@@ -6,7 +6,8 @@
 
     for i in range(len(weight_by_node) - 1):
         for (u, v), weight in weight_by_edge.items():
-            weight_by_edge[u, v] = min(
+            weight_by_node[v] = min(
+                weight_by_node[v],
                 weight_by_node[u] + weight,
                 weight_by_node[v]
             )
