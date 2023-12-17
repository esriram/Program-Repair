--- 
+++ 
@@ -18,6 +18,7 @@
     _currInputRowStart = start - (_inputEnd - _currInputRowStart);
 
     // And then update buffer settings
+    _currBufferStart = start;
     _inputBuffer = buf;
     _inputPtr = start;
     _inputEnd = end;
