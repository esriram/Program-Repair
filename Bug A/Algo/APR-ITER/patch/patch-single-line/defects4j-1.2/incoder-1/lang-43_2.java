--- 
+++ 
@@ -3,6 +3,8 @@
     int start = pos.getIndex();
     char[] c = pattern.toCharArray();
     if (escapingOn && c[start] == QUOTE) {
+        // The escape character is followed by another character so move back
+        next(pos);
         return appendTo == null ? null : appendTo.append(QUOTE);
     }
     int lastHold = start;
