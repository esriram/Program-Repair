--- 
+++ 
@@ -3,6 +3,9 @@
     Node tortoise = node;
 
     while (true) {
+        if (hare == null)
+            return false;
+
         if (hare.getSuccessor() == null)
             return false;
 
