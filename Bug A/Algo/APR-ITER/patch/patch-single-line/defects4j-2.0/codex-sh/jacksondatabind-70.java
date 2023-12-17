--- 
+++ 
@@ -12,7 +12,10 @@
         if (!found) {
             // 09-Jan-2017, tatu: Important: must check name slot and NOT property name,
             //   as only former is lower-case in case-insensitive case
-            found = key.equals(prop.getName());
+            String name = (String) _hashArea[i-1];
+            if (name.equals(key)) {
+                found = true;
+            }
             if (found) {
                 // need to leave a hole here
                 _propsInOrder[_findFromOrdered(prop)] = null;
