--- 
+++ 
@@ -1,5 +1,6 @@
 public static int gcd(int u, int v) {
-    if (u * v == 0) {
+    // This implementation is adapted from Sedgewick's algorithm.
+    if (u == 0 || v == 0) {
         return (Math.abs(u) + Math.abs(v));
     }
     // keep u and v negative, as negative integers range down to
