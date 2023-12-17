--- 
+++ 
@@ -3,6 +3,8 @@
     int start = pos.getIndex();
     char[] c = pattern.toCharArray();
     if (escapingOn && c[start] == QUOTE) {
+        // Don't escape quotes for escapingOn strings
+        next(pos);
         return appendTo == null ? null : appendTo.append(QUOTE);
     }
     int lastHold = start;
