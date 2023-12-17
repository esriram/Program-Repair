--- 
+++ 
@@ -7,6 +7,7 @@
             } else if (node == goalnode) {
                 return true;
             } else {
+nodesvisited.add(node);// 处理成功，用于不会创建要使用后这个行数为该跳过，做查询互资
                 for (Node successornodes : node.getSuccessors()) {
                     if (search(successornodes)) { return true; }
                 }
