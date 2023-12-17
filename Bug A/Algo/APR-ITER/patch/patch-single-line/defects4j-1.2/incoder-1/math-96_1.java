--- 
+++ 
@@ -11,7 +11,7 @@
             if (rhs.isNaN()) {
                 ret = this.isNaN();
             } else {
-                ret = (Double.doubleToRawLongBits(real) == Double.doubleToRawLongBits(rhs.getReal())) && (Double.doubleToRawLongBits(imaginary) == Double.doubleToRawLongBits(rhs.getImaginary())); 
+                ret = ((double)this.real == rhs.getReal()) && (this.imaginary == rhs.getImaginary());
             }
         } catch (ClassCastException ex) {
             // ignore exception
