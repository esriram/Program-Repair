--- 
+++ 
@@ -17,7 +17,11 @@
             }
 
             double sumWts = 0;
-            for (int i = 0; i < weights.length; i++) {
+            double sumWts2 = 0.0;
+            for (int i = begin; i < begin + length; i++) {
+                dev = values[i] - mean;
+                accum2 += weights[i] * dev;
+                sumWts2 += weights[i] * dev;
                 sumWts += weights[i];
             }
 
