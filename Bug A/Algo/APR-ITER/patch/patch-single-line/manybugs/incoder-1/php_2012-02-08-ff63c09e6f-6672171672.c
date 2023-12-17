--- 
+++ 
@@ -78,7 +78,7 @@
 
 				if (track_vars_array) {
 					ht = Z_ARRVAL_P(track_vars_array);
-					zend_hash_del(ht, var, var_len + 1);
+					zend_symtable_del(ht, var,1+(var_len));
 				}
 
 				zval_dtor(val);
