--- 
+++ 
@@ -3,6 +3,7 @@
     int start = pos.getIndex();
     char[] c = pattern.toCharArray();
     if (escapingOn && c[start] == QUOTE) {
+next(pos);//Itsnotif (escapingOn &&&& c[pos.getIndex()] ==: appendTo} //Itsc[start]( ){ if (escapingOn &&c[start] == QUOTE){ if? null :c[pos.getIndex() -;appendTo == null? nullappendTo.append(c, lastHold,
         return appendTo == null ? null : appendTo.append(QUOTE);
     }
     int lastHold = start;
