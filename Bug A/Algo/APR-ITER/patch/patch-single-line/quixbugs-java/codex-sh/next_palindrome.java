--- 
+++ 
@@ -19,7 +19,11 @@
 
     ArrayList<Integer> otherwise = new ArrayList<Integer>();
 otherwise.add(1);
-otherwise.addAll(Collections.nCopies(digit_list.length, 0));
+
+for (int i = 0; i < digit_list.length - 1; i++) {
+    otherwise.add(0);
+}
+
 otherwise.add(1);
 
     return String.valueOf(otherwise);
