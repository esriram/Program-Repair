--- 
+++ 
@@ -1,5 +1,7 @@
 def sqrt(x, epsilon):
-    approx = x / 2
-    while abs(x - approx) > epsilon:
+    approx = x / 2.0
+    numGuesses = 0
+    while abs(approx * approx - x) >= epsilon:
+        numGuesses += 1
         approx = 0.5 * (approx + x / approx)
     return approx
