--- 
+++ 
@@ -14,7 +14,7 @@
             if (i == 0 || j == 0) {
                 memo[i][j] = 0;
             }
-            else if (weight < j) {
+            else if (weight <= j) {
                 memo[i][j] = Math.max(memo[i - 1][j], value + memo[i - 1][j - weight]);
             }
             else {
