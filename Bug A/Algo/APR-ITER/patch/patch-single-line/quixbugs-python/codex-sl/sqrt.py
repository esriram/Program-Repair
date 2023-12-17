--- 
+++ 
@@ -1,5 +1,5 @@
 def sqrt(x, epsilon):
     approx = x / 2
-    while abs(x - approx) > epsilon:
+    while abs(x - approx * approx) > epsilon:
         approx = 0.5 * (approx + x / approx)
     return approx
