--- 
+++ 
@@ -5,6 +5,9 @@
     if (total < 0) {
         return 0;
     }
+    if (coins.length == 0) {
+        return 0;
+    }
 
     int first = coins[0];
     int[] rest = Arrays.copyOfRange(coins, 1, coins.length);
