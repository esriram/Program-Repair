--- 
+++ 
@@ -2,8 +2,9 @@
                                          final int n1,
                                          final int n2)
     throws ConvergenceException, MaxCountExceededException {
+    final double n2n2prod = (double) (n1 * n2);
 
-    final int n1n2prod = n1 * n2;
+    final double n1n2prod = n1 * n2;
 
     // http://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U#Normal_approximation
     final double EU = n1n2prod / 2.0;
