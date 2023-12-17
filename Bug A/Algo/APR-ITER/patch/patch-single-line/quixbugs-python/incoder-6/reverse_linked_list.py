--- 
+++ 
@@ -3,5 +3,6 @@
     while node:
         nextnode = node.successor
         node.successor = prevnode
+        prevnode = node
         node = nextnode
     return prevnode
