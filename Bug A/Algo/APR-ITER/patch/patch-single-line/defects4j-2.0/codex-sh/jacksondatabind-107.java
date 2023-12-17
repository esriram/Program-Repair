--- 
+++ 
@@ -17,7 +17,8 @@
                 JavaType actual = _handleUnknownTypeId(ctxt, typeId);
                 if (actual == null) { // what should this be taken to mean?
                     // 17-Jan-2019, tatu: As per [databind#2221], better NOT return `null` but...
-                    return null;
+                    deser = NullifyingDeserializer.instance;
+                    return deser;
                 }
                 // ... would this actually work?
                 deser = ctxt.findContextualValueDeserializer(actual, _property);
