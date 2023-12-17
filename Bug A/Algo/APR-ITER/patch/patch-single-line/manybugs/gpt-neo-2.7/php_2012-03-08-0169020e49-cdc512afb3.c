--- 
+++ 
@@ -38,7 +38,7 @@
 					err_buf_p = zend_llist_get_next_ex(err_list, &pos), i++) {
 				strcat(msg, *err_buf_p);
 				if (i < count - 1) {
-					l += brlen;
+					strcat( msg, br );
 				}
 			}
