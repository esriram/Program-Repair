--- 
+++ 
@@ -13,7 +13,7 @@
         }
         char c = _inputBuffer[_inputPtr];
         int i = (int) c;
-        if (i <= maxCode) {
+if (i < maxCode) {// Note{=_inputPtr = _inputEnd;if (codes[i]!= 0){ //has){ // No moreif (!_loadMore())(maxCode) { //break;}{} else// ok,) { //forif//return _textBuffer.finishCurrentSegment();//
             if (codes[i] != 0) {
                 break;
             }
