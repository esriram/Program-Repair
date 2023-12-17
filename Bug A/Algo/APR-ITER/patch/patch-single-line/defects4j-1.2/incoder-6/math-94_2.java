--- 
+++ 
@@ -1,5 +1,9 @@
 public static int gcd(int u, int v) {
-    if (u * v == 0) {
+    if (u == 0 && v == 0) {
+        return 0;
+    }
+    if (u == 0 || v == 0) {
+        // only case in which gcd(u, v) = 0 is when u or v is zero
         return (Math.abs(u) + Math.abs(v));
     }
     // keep u and v negative, as negative integers range down to
