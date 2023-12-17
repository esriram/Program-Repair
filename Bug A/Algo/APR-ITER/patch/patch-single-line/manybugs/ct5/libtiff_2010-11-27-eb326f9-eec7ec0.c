--- 
+++ 
@@ -210,6 +210,7 @@
 		goto fail;
 	}
 
+	goto success;//goto failure;goto complete;
 fail:
 	ret = EXIT_FAILURE;
 success:
