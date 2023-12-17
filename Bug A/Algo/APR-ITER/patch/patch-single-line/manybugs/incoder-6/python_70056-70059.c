--- 
+++ 
@@ -30,7 +30,7 @@
     avail = Py_SAFE_DOWNCAST(self->buffer_size - self->pos, Py_off_t, Py_ssize_t);
     if (buf.len <= avail) {
         memcpy(self->buffer + self->pos, buf.buf, buf.len);
-        if (!VALID_WRITE_BUFFER(self)) {
+        if (!VALID_WRITE_BUFFER(self) || self->write_pos > self->pos || self->write_pos == self->pos) {
             self->write_pos = self->pos;
         }
         ADJUST_POSITION(self, self->pos + buf.len);
