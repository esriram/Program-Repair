--- 
+++ 
@@ -5,7 +5,7 @@
     nodesseen = set()
     nodesseen.add(startnode)
 
-    while True:
+    while len(queue) > 0:
         node = queue.popleft()
 
         if node is goalnode:
