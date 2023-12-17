--- 
+++ 
@@ -13,7 +13,8 @@
         }
         char c = _inputBuffer[_inputPtr];
         int i = (int) c;
-        if (i <= maxCode) {
+        if (i < maxCode) {
+            // We will skip first character, so can ignore its code
             if (codes[i] != 0) {
                 break;
             }
