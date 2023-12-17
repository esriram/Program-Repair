--- 
+++ 
@@ -18,7 +18,7 @@
     while (true)
     {
         text = padding + text.substring(pos).trim();
-        pos = findWrapPos(text, width, nextLineTabStop);
+        pos = findWrapPos(text, width, 0);
 
         if (pos == -1)
         {
