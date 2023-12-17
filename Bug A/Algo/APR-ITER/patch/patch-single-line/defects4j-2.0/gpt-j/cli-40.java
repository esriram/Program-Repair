--- 
+++ 
@@ -38,6 +38,6 @@
     }
     else
     {
-        return null;
+        throw new ParseException("unknown type: " + clazz);
     }
 }
