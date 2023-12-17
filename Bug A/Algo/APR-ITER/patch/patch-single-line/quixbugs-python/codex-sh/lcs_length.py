--- 
+++ 
@@ -5,6 +5,8 @@
     for i in range(len(s)):
         for j in range(len(t)):
             if s[i] == t[j]:
-                dp[i, j] = dp[i - 1, j] + 1
+                dp[i, j] = dp[i - 1, j - 1] + 1
+
+    print(dp)
 
     return max(dp.values()) if dp else 0
