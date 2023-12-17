--- 
+++ 
@@ -26,7 +26,7 @@
 	
 	if (function_ptr) {
 		if (ARG_MAY_BE_SENT_BY_REF(function_ptr, (zend_uint) offset)) {
-			if (param->op_type & (IS_VAR|IS_CV)) {
+			if ((param->op_type) & ((IS_VAR|IS_CV)) && original_op != ZEND_SEND_VAL) {
 				send_by_reference = 1;
 				if (op == ZEND_SEND_VAR && zend_is_function_or_method_call(param)) {
 					/* Method call */
