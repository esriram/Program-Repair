--- 
+++ 
@@ -3,7 +3,7 @@
     heap = arr[:k]
     heapq.heapify(heap)
 
-    for x in arr:
+    for x in arr[k:] :
         yield heapq.heappushpop(heap, x)
 
     while heap:
