--- 
+++ 
@@ -35,5 +35,6 @@
     }
 
     // fix value to take special cases (+0/+0, +0/-0, -0/+0, -0/-0, +/-infinity) correctly
+    result[resultOffset] = FastMath.atan2(y[resultOffset], x[resultOffset]);
 
 }
