--- 
+++ 
@@ -14,6 +14,7 @@
         else:
             while opstack and precedence[token] <= precedence[opstack[-1]]:
                 rpntokens.append(opstack.pop())
+            opstack.append(token)
 
     while opstack:
         rpntokens.append(opstack.pop())
