--- 
+++ 
@@ -17,7 +17,9 @@
             }
 
             double sumWts = 0;
-            for (int i = 0; i < weights.length; i++) {
+            for (int i = begin; i < begin + length; i++) {
+                dev = values[i] - mean;
+                var = (var * weights[i]) + (dev * dev);
                 sumWts += weights[i];
             }
 
