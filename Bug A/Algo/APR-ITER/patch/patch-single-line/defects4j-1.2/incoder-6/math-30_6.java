--- 
+++ 
@@ -2,8 +2,8 @@
                                          final int n1,
                                          final int n2)
     throws ConvergenceException, MaxCountExceededException {
-
-    final int n1n2prod = n1 * n2;
+    // http://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U#Critical_point_and_critical_point_scaling_factor
+    final double n1n2prod = n1 * n2;
 
     // http://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U#Normal_approximation
     final double EU = n1n2prod / 2.0;
