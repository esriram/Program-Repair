--- 
+++ 
@@ -1,7 +1,14 @@
 public static ArrayList<Integer> sieve(Integer max) {
     ArrayList<Integer> primes = new ArrayList<Integer>();
     for (int n=2; n<max+1; n++) {
-        if (any(list_comp(n, primes))) {
+        boolean prime = true;
+        for (int i = 2; i < n; i++) {
+            if (n % i == 0) {
+                prime = false;
+                break;
+            }
+        }
+        if (prime) {
             primes.add(n);
         }
     }
