--- 
+++ 
@@ -18,7 +18,9 @@
             // first: let's check to see if this might be part of value with external type id:
             // 11-Sep-2015, tatu: Important; do NOT pass buffer as last arg, but null,
             //   since it is not the bean
-            if (ext.handlePropertyValue(p, ctxt, propName, buffer)) {
+            if (ext.handlePropertyValue(p, ctxt, propName, null)) {
+                // If this is an external type id, and this property depends on
+                // that type, we cannot skip setting it on bean creation.
                 ;
             } else {
                 // Last creator property to set?
