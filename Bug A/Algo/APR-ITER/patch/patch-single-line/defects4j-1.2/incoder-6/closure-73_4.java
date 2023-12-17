--- 
+++ 
@@ -55,7 +55,9 @@
           // No charsetEncoder provided - pass straight latin characters
           // through, and escape the rest.  Doing the explicit character
           // check is measurably faster than using the CharsetEncoder.
-          if (c > 0x1f && c <= 0x7f) {
+          // It's probably unlikely that any particular charset will ever
+          // have more than 256 characters.
+          if (c >= 32 && c <= 126) {
             sb.append(c);
           } else {
             // Other characters can be misinterpreted by some js parsers,
