--- 
+++ 
@@ -78,7 +78,7 @@
 
 				if (track_vars_array) {
 					ht = Z_ARRVAL_P(track_vars_array);
-					zend_hash_del(ht, var, var_len + 1);
+					zend_symtable_del(ht, var, var_len + 1);
 				}
 
 				zval_dtor(val);
