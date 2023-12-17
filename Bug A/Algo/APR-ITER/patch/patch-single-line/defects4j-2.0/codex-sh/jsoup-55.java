--- 
+++ 
@@ -12,6 +12,7 @@
             break;
         default:
             t.error(this);
+            r.unconsume();
             t.transition(BeforeAttributeName);
     }
 }
