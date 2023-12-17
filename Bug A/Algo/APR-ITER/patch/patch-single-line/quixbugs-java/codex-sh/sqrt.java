--- 
+++ 
@@ -1,7 +1,7 @@
 public static double sqrt(double x, double epsilon) {
-    double approx = x / 2f;
-    while (Math.abs(x-approx) > epsilon) {
-        approx = 0.5f * (approx + x / approx);
+    double approx = x / 2;
+    while (Math.abs(approx * approx - x) > epsilon) {
+        approx = (approx + x / approx) / 2;
     }
     return approx;
 }
