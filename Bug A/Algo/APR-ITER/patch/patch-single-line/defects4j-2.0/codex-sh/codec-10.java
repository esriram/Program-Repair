--- 
+++ 
@@ -23,8 +23,8 @@
     txt = txt.replaceAll("^trough", "trou2f");  // 2.0 only - note the spec says ^enough here again, c+p error I assume
     txt = txt.replaceAll("^gn", "2n");
 
-    // End 
-    txt = txt.replaceAll("^mb", "m2");
+    // End of string fixes
+    txt = txt.replaceAll("mb$", "m2");          // 2.0 only
 
     // 4. Handle replacements
     txt = txt.replaceAll("cq", "2q");
