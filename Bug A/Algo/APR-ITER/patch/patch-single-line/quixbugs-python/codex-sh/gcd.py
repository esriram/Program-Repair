--- 
+++ 
@@ -2,4 +2,11 @@
     if b == 0:
         return a
     else:
-        return gcd(a % b, b)
+        return gcd(b, a % b)
+
+
+if __name__ == '__main__':
+    a, b = input().split()
+    a = int(a)
+    b = int(b)
+    print(gcd(a, b))
