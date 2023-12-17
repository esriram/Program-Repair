--- 
+++ 
@@ -17,7 +17,8 @@
             }
 
             double sumWts = 0;
-            for (int i = 0; i < weights.length; i++) {
+            double sum = 0.0;
+            for (int i = begin; i < begin + length; i++) {
                 sumWts += weights[i];
             }
 
