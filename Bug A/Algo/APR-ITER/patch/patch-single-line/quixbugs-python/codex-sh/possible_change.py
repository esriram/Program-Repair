--- 
+++ 
@@ -1,7 +1,7 @@
 def possible_change(coins, total):
     if total == 0:
         return 1
-    if total < 0:
+    if total < 0 or not coins:
         return 0
 
     first, *rest = coins
