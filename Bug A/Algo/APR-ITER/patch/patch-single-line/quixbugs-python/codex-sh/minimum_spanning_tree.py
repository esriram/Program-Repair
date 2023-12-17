--- 
+++ 
@@ -8,6 +8,6 @@
             mst_edges.add(edge)
             group_by_node[u].update(group_by_node[v])
             for node in group_by_node[v]:
-                group_by_node[node].update(group_by_node[u])
+                group_by_node[node] = group_by_node[u]
 
     return mst_edges
