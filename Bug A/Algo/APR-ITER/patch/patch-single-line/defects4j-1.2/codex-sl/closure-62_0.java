--- 
+++ 
@@ -30,7 +30,7 @@
     // charno == sourceExpert.length() means something is missing
     // at the end of the line
     if (excerpt.equals(LINE)
-        && 0 <= charno && charno < sourceExcerpt.length()) {
+        && 0 <= charno && charno <= sourceExcerpt.length()) {
       for (int i = 0; i < charno; i++) {
         char c = sourceExcerpt.charAt(i);
         if (Character.isWhitespace(c)) {