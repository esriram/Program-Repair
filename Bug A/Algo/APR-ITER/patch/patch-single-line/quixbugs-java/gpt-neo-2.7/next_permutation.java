--- 
+++ 
@@ -2,7 +2,7 @@
     for (int i=perm.size()-2; i!=-1; i--) {
         if (perm.get(i) < perm.get(i+1)) {
             for (int j=perm.size()-1; j!=i; j--) {
-                if (perm.get(j) < perm.get(i)) {
+                if (perm.get(i) < perm.get(j)) {
                     ArrayList<Integer> next_perm = perm;
                     int temp_j = perm.get(j);
                     int temp_i = perm.get(i);
