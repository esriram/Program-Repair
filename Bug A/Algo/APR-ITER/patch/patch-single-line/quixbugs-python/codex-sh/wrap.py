--- 
+++ 
@@ -6,6 +6,5 @@
             end = cols
         line, text = text[:end], text[end:]
         lines.append(line)
-
-    {}
+    lines.append(text)
     return lines
