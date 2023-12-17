--- 
+++ 
@@ -2,7 +2,7 @@
     hare = tortoise = node
 
     while True:
-        if hare.successor is None:
+        if hare is None or hare.successor is None:
             return False
 
         tortoise = tortoise.successor
