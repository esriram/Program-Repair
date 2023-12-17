--- 
+++ 
@@ -1,5 +1,5 @@
 def sqrt(x, epsilon):
     approx = x / 2
-    while abs(x - approx) > epsilon:
+    while abs(approx ** 2 - x) > epsilon:
         approx = 0.5 * (approx + x / approx)
     return approx
