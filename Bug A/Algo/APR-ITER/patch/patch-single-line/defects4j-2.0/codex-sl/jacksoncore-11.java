--- 
+++ 
@@ -6,6 +6,7 @@
         _hashShared = false;
         // 09-Sep-2015, tatu: As per [jackson-core#216], also need to ensure
         //    we rehash as needed, as need-rehash flag is not copied from parent
+        _verifyNeedForRehash();
     }
     if (_needRehash) {
         rehash();
