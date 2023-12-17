--- 
+++ 
@@ -3,6 +3,7 @@
     int start = pos.getIndex();
     char[] c = pattern.toCharArray();
     if (escapingOn && c[start] == QUOTE) {
+        next(pos);
         return appendTo == null ? null : appendTo.append(QUOTE);
     }
     int lastHold = start;
