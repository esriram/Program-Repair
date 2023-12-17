--- 
+++ 
@@ -1,6 +1,6 @@
 def subsequences(a, b, k):
     if k == 0:
-        return []
+        return [[]]
 
     ret = []
     for i in range(a, b + 1 - k):
