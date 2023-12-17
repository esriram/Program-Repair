--- 
+++ 
@@ -38,6 +38,6 @@
     }
     else
     {
-        return null;
+        throw new ParseException("Cannot create value of type " + clazz.getName());
     }
 }
