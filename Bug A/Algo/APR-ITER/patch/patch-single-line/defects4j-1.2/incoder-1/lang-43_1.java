--- 
+++ 
@@ -3,6 +3,8 @@
     int start = pos.getIndex();
     char[] c = pattern.toCharArray();
     if (escapingOn && c[start] == QUOTE) {
+        // A quoted string that doesn't escape
+        next(pos);
         return appendTo == null ? null : appendTo.append(QUOTE);
     }
     int lastHold = start;
