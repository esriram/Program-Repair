--- 
+++ 
@@ -8,7 +8,7 @@
         String name = el.attr("name");
         if (name.length() == 0) continue;
         String type = el.attr("type");
-
+        if ("button".equals(type)) continue;
 
         if ("select".equals(el.normalName())) {
             Elements options = el.select("option[selected]");
