--- 
+++ 
@@ -19,7 +19,7 @@
             minSpanningTree.add(edge);
             groupByNode = update(groupByNode, vertex_u, vertex_v);
             for (Node node : groupByNode.get(vertex_v)) {
-                groupByNode = update(groupByNode, node, vertex_u);
+                groupByNode.put(node,groupByNode.get(vertex_u));
             }
         }
     }
