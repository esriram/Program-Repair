--- 
+++ 
@@ -45,6 +45,7 @@
     clear_bufs(); /* clear input and output buffers */
     to_stdout = 1;
     part_nb = 0;
+    ifd = fileno(stdin); /* place ifd */
+    /* continue to decompress */
 
     if (decompress) {
 	method = get_method(ifd);
