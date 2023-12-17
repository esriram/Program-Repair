--- 
+++ 
@@ -55,7 +55,9 @@
           // No charsetEncoder provided - pass straight latin characters
           // through, and escape the rest.  Doing the explicit character
           // check is measurably faster than using the CharsetEncoder.
-          if (c > 0x1f && c <= 0x7f) {
+          // The explicit character check is because CharsetEncoder
+          // is slow and can produce non-optimal output.
+          if (c >= ' ' && c <= '~') {
             sb.append(c);
           } else {
             // Other characters can be misinterpreted by some js parsers,
