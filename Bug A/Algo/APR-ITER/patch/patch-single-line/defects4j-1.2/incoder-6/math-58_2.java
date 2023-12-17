--- 
+++ 
@@ -1,4 +1,5 @@
 public double[] fit() {
     final double[] guess = (new ParameterGuesser(getObservations())).guess();
-    return fit(new Gaussian.Parametric(), guess);
+    final double[] fitted = this.fit(guess);
+    return fitted;
 }
