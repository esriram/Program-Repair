--- 
+++ 
@@ -1,5 +1,5 @@
 public static ArrayList<Integer> mergesort(ArrayList<Integer> arr) {
-    if (arr.size() == 0) { // <= 1 in correct version
+		if (arr.size() <= 1) {
         return arr;
     } else {
         int middle = arr.size() / 2;
