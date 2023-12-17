--- 
+++ 
@@ -3,7 +3,7 @@
     Node tortoise = node;
 
     while (true) {
-        if (hare.getSuccessor() == null)
+        if (hare == null || hare.getSuccessor() == null)
             return false;
 
         tortoise = tortoise.getSuccessor();
