--- 
+++ 
@@ -2,8 +2,8 @@
                                          final int n1,
                                          final int n2)
     throws ConvergenceException, MaxCountExceededException {
-
-    final int n1n2prod = n1 * n2;
+    // http://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U#Computing_the_U%27_of_the_statistic
+    final double n1n2prod = n1 * n2;
 
     // http://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U#Normal_approximation
     final double EU = n1n2prod / 2.0;
