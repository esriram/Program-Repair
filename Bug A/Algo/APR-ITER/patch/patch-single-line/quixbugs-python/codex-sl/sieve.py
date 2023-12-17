--- 
+++ 
@@ -1,6 +1,6 @@
 def sieve(max):
     primes = []
     for n in range(2, max + 1):
-        if any(n % p > 0 for p in primes):
+        if all(n % p for p in primes):
             primes.append(n)
     return primes
