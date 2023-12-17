--- 
+++ 
@@ -37,7 +37,7 @@
                     if (escapeMode != EscapeMode.xhtml)
                         accum.append("&nbsp;");
                     else
-                        accum.append(c);
+                        accum.append("&#xa0;");
                     break;
                 case '<':
                     if (!inAttribute)
