--- 
+++ 
@@ -5,7 +5,7 @@
 	const char gz_magic[] = "\x1f\x8b\x08";
 	const char bz_magic[] = "BZh";
 	char *pos, test = '\0';
-	const int window_size = 1024 + sizeof(token);
+	const int window_size = 1024; // size of window
 	char buffer[1024 + sizeof(token)]; /* a 1024 byte window + the size of the halt_compiler token (moving window) */
 	const long readsize = sizeof(buffer) - sizeof(token);
 	const long tokenlen = sizeof(token) - 1;
