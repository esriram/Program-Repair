--- 
+++ 
@@ -19,7 +19,8 @@
             minSpanningTree.add(edge);
             groupByNode = update(groupByNode, vertex_u, vertex_v);
             for (Node node : groupByNode.get(vertex_v)) {
-                groupByNode = update(groupByNode, node, vertex_u);
+                //System.out.printf("u: %s, v: %s weight: %d\n", vertex_u.getValue(), node.getValue(), edge.weight);
+                groupByNode.put(node, groupByNode.get(vertex_u));
             }
         }
     }
