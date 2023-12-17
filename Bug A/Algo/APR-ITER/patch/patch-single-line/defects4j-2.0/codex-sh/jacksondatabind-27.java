--- 
+++ 
@@ -17,8 +17,8 @@
         if (creatorProp != null) {
             // first: let's check to see if this might be part of value with external type id:
             // 11-Sep-2015, tatu: Important; do NOT pass buffer as last arg, but null,
-            //   since it is not the bean
-            if (ext.handlePropertyValue(p, ctxt, propName, buffer)) {
+            //   since it is not the bean instance. This matters for JsonDeserializer.
+            if (ext.handlePropertyValue(p, ctxt, propName, null)) {
                 ;
             } else {
                 // Last creator property to set?
