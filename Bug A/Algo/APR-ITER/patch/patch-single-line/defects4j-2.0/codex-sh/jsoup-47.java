--- 
+++ 
@@ -40,8 +40,8 @@
                         accum.append("&#xa0;");
                     break;
                 case '<':
-                    // escape when in character data or when in a xml attribue val; not needed in html attr val
-                    if (!inAttribute)
+                    // escape when in character data or when in a xml attribue val; not needed in html attr val or script/style
+                    if (!inAttribute || escapeMode == EscapeMode.xhtml)
                         accum.append("&lt;");
                     else
                         accum.append(c);
