--- 
+++ 
@@ -1,7 +1,14 @@
 public static ArrayList<Integer> sieve(Integer max) {
     ArrayList<Integer> primes = new ArrayList<Integer>();
     for (int n=2; n<max+1; n++) {
-        if (any(list_comp(n, primes))) {
+        boolean prime = true;
+        for (int p : primes) {
+            if (n % p == 0) {
+                prime = false;
+                break;
+            }
+        }
+        if (prime) {
             primes.add(n);
         }
     }
