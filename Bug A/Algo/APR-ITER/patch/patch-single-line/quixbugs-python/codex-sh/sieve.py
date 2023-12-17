--- 
+++ 
@@ -1,6 +1,9 @@
 def sieve(max):
     primes = []
     for n in range(2, max + 1):
-        if any(n % p > 0 for p in primes):
+        for p in primes:
+            if n % p == 0:
+                break
+        else:
             primes.append(n)
     return primes
