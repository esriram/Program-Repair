--- 
+++ 
@@ -35,5 +35,6 @@
     }
 
     // fix value to take special cases (+0/+0, +0/-0, -0/+0, -0/-0, +/-infinity) correctly
+    result[resultOffset + 0] = FastMath.atan2(y[yOffset], x[xOffset]);
 
 }
