--- 
+++ 
@@ -17,7 +17,7 @@
             Double b = (Double) stack.pop();
     Double c = 0.0;
     BinaryOperator<Double> bin_op = op.get(token);
-    c = bin_op.apply(a,b);
+            c = bin_op.apply(b, a);
             stack.push(c);
         }
     }
