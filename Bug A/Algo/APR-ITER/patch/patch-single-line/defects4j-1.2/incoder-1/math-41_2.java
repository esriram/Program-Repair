--- 
+++ 
@@ -17,7 +17,9 @@
             }
 
             double sumWts = 0;
-            for (int i = 0; i < weights.length; i++) {
+            double sum = 0;
+            for (int i = begin; i < begin + length; i++) {
+                sum += weights[i];
                 sumWts += weights[i];
             }
 
