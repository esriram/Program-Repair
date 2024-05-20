--- 
+++ 
@@ -29,8 +29,8 @@
     // padding equal to the excerpt and arrow at the end
     // charno == sourceExpert.length() means something is missing
     // at the end of the line
-    if (excerpt.equals(LINE)
-        && 0 <= charno && charno < sourceExcerpt.length()) {
+    if (excerpt.equals(LINE) && (0 <= charno) &&
+        (charno <= sourceExcerpt.length())) {
       for (int i = 0; i < charno; i++) {
         char c = sourceExcerpt.charAt(i);
         if (Character.isWhitespace(c)) {
