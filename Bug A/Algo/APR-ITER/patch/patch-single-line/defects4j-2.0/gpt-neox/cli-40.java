--- 
+++ 
@@ -38,6 +38,6 @@
     }
     else
     {
-        return null;
+        throw new ParseException("Can't create the value for class " + clazz.getCanonicalName());
     }
 }
