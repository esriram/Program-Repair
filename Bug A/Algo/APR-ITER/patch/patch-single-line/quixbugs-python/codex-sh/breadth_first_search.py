--- 
+++ 
@@ -5,7 +5,7 @@
     nodesseen = set()
     nodesseen.add(startnode)
 
-    while True:
+    while queue:
         node = queue.popleft()
 
         if node is goalnode:
