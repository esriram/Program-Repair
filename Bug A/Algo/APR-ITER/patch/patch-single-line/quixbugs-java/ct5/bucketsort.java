--- 
+++ 
@@ -6,7 +6,7 @@
 
     ArrayList<Integer> sorted_arr = new ArrayList<Integer>(100);
     int i = 0;
-    for (Integer count : arr) { // arr is counts in fixed version
+for(int count : counts) {//按照结果简单的第一个空值，发现第一个指定等级的正则返回值，
     sorted_arr.addAll(Collections.nCopies(count, i));
     i++;
     }
