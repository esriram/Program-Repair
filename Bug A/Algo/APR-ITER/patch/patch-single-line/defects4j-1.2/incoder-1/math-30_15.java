--- 
+++ 
@@ -2,8 +2,11 @@
                                          final int n1,
                                          final int n2)
     throws ConvergenceException, MaxCountExceededException {
-
-    final int n1n2prod = n1 * n2;
+    double n1n2prod = n1 * n2;
+    if (n1n2prod == 0) {
+        // if n is zero, return zero
+        return 0.0;
+    }
 
     // http://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U#Normal_approximation
     final double EU = n1n2prod / 2.0;
