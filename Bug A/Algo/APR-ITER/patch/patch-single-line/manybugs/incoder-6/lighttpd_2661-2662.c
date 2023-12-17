--- 
+++ 
@@ -1,6 +1,7 @@
 static void accesslog_append_escaped(buffer *dest, buffer *str) {
 	/* replaces non-printable chars with \xHH where HH is the hex representation of the byte */
 	/* exceptions: " => \", \ => \\, whitespace chars => \n \t etc. */
+	if (str->used == 0 || str == NULL) {
+		return;
+	}
 	buffer_prepare_append(dest, str->used - 1);
 
 	for (unsigned int i = 0; i < str->used - 1; i++) {
