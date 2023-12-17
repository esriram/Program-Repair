--- 
+++ 
@@ -30,7 +30,10 @@
                 }
                 break;
         }
-        if (lineLength > 0) {
+        if (lineLength > 0 && pos > 0) {
+            // If we have any chars in our buffer we always dump them and
+            // reset the current line length to 0, to prevent pre-mature line
+            // ending for final chunk.
             System.arraycopy(lineSeparator, 0, buf, pos, lineSeparator.length);
             pos += lineSeparator.length;
         }
