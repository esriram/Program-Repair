--- 
+++ 
@@ -13,7 +13,7 @@
         }
         char c = _inputBuffer[_inputPtr];
         int i = (int) c;
-        if (i <= maxCode) {
+        if (i < maxCode) {
             if (codes[i] != 0) {
                 break;
             }
