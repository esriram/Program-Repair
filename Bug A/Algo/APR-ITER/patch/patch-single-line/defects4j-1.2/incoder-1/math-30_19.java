--- 
+++ 
@@ -2,8 +2,15 @@
                                          final int n1,
                                          final int n2)
     throws ConvergenceException, MaxCountExceededException {
+    if (n1 <= 0 || n2 <= 0) {
+        throw new IllegalArgumentException("N1 and N2 must be positive");
+    }
 
-    final int n1n2prod = n1 * n2;
+    if (n1 == 1 && n2 == 1) {
+        return 1;
+    }
+
+    final double n1n2prod = n1 * n2;
 
     // http://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U#Normal_approximation
     final double EU = n1n2prod / 2.0;
