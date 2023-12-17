--- 
+++ 
@@ -17,7 +17,8 @@
             }
 
             double sumWts = 0;
-            for (int i = 0; i < weights.length; i++) {
+            boolean isBiasCorrected = true;
+            for (int i = begin; i < begin + length; i++) {
                 sumWts += weights[i];
             }
 
