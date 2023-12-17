--- 
+++ 
@@ -17,6 +17,7 @@
             while (!opstack.isEmpty() && precedence.get(operator) <= precedence.get(opstack.getLast())) {
                 rpntokens.add(opstack.pop());
             }
+            opstack.push(token);
         }
 
     }
