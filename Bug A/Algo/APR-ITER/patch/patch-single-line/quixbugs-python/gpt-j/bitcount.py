--- 
+++ 
@@ -1,6 +1,6 @@
 def bitcount(n):
     count = 0
     while n:
-        n ^= n - 1
+        n &= n-1
         count += 1
     return count
