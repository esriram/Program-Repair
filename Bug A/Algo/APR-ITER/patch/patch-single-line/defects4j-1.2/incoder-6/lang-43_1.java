--- 
+++ 
@@ -3,6 +3,10 @@
     int start = pos.getIndex();
     char[] c = pattern.toCharArray();
     if (escapingOn && c[start] == QUOTE) {
+        next(pos);
+        if (appendTo == null) {
+            return null;
+        }
         return appendTo == null ? null : appendTo.append(QUOTE);
     }
     int lastHold = start;
