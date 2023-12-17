--- 
+++ 
@@ -1,3 +1,8 @@
 public static float max(final float a, final float b) {
-    return (a <= b) ? b : (Float.isNaN(a + b) ? Float.NaN : b);
+		if (Float.isNaN(a)) {
+			return Float.NaN;
+		} else if (Float.isNaN(b)) {
+			return Float.NaN;
+		}
+		return Math.max(a, b);
 }
