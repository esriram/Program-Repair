--- 
+++ 
@@ -55,7 +55,7 @@
           // No charsetEncoder provided - pass straight latin characters
           // through, and escape the rest.  Doing the explicit character
           // check is measurably faster than using the CharsetEncoder.
-          if (c > 0x1f && c <= 0x7f) {
+          if (c > 0x1F && c < 0x7F) {
             sb.append(c);
           } else {
             // Other characters can be misinterpreted by some js parsers,
