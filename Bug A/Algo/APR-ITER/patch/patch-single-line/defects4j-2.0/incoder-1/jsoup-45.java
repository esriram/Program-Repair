--- 
+++ 
@@ -10,7 +10,10 @@
         if ("select".equals(name)) {
             transition(HtmlTreeBuilderState.InSelect);
             break; // frag
-        } else if (("td".equals(name) || "td".equals(name) && !last)) {
+        } else if ("td".equals(name)) {
+            transition(HtmlTreeBuilderState.InCell);
+            break;
+        } else if ("th".equals(name)) {
             transition(HtmlTreeBuilderState.InCell);
             break;
         } else if ("tr".equals(name)) {
