--- 
+++ 
@@ -36,7 +36,8 @@
             {
                 // if the value is not yes, true or 1 then don't add the
                 // option to the CommandLine
-                break;
+                // otherwise we add it to the options list
+                continue;
             }
 
             cmd.addOption(opt);
