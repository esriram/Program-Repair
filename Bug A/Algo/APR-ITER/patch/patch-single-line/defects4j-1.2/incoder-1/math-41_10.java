--- 
+++ 
@@ -17,7 +17,9 @@
             }
 
             double sumWts = 0;
-            for (int i = 0; i < weights.length; i++) {
+            for (int i = begin + length - 1; i >= begin; i--) {
+                dev = values[i] - mean;
+                accum += weights[i] * dev;
                 sumWts += weights[i];
             }
 
