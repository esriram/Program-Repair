--- 
+++ 
@@ -4,5 +4,5 @@
     while num > 0:
         i = num % b
         num = num // b
-        result = result + alphabet[i]
+        result = alphabet[i] + result
     return result
