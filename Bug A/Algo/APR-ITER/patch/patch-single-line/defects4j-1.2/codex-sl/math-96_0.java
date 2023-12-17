--- 
+++ 
@@ -11,7 +11,7 @@
             if (rhs.isNaN()) {
                 ret = this.isNaN();
             } else {
-                ret = (Double.doubleToRawLongBits(real) == Double.doubleToRawLongBits(rhs.getReal())) && (Double.doubleToRawLongBits(imaginary) == Double.doubleToRawLongBits(rhs.getImaginary())); 
+                ret = (real == rhs.real) && (imaginary == rhs.imaginary);
             }
         } catch (ClassCastException ex) {
             // ignore exception
