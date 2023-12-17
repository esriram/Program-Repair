--- 
+++ 
@@ -7,5 +7,5 @@
         line, text = text[:end], text[end:]
         lines.append(line)
 
-    {}
+    lines.append(text)
     return lines
