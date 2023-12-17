--- 
+++ 
@@ -2,4 +2,10 @@
     if b == 0:
         return a
     else:
-        return gcd(a % b, b)
+        return gcd(b, a % b)
+
+# Test cases
+print(gcd(12, 14))
+print(gcd(6, 12))
+print(gcd(1, 1))
+
