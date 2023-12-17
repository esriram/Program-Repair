--- 
+++ 
@@ -1,5 +1,6 @@
 public static int gcd(int u, int v) {
-    if (u * v == 0) {
+    /* assert u>=0 && v>=0; */
+    if (v == 0 || u == 0) {
         return (Math.abs(u) + Math.abs(v));
     }
     // keep u and v negative, as negative integers range down to
