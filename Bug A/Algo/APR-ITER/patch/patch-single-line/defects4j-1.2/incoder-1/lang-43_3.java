--- 
+++ 
@@ -3,6 +3,9 @@
     int start = pos.getIndex();
     char[] c = pattern.toCharArray();
     if (escapingOn && c[start] == QUOTE) {
+        // The last char of the string has been escaped by a special escape
+        // sequence (QUOTE).
+        next(pos);
         return appendTo == null ? null : appendTo.append(QUOTE);
     }
     int lastHold = start;
