--- 
+++ 
@@ -16,8 +16,7 @@
     for (int k = 0; k < numNodes; k++) {
         for (int i = 0; i < numNodes; i++) {
             for (int j = 0; j < numNodes; j++) {
-                int update_length = Math.min(length_by_path.get(Arrays.asList(i,j)),
-                        length_by_path.get(Arrays.asList(i,k)) + length_by_path.get(Arrays.asList(j,k)));
+                int update_length = Math.min(length_by_path.get(Arrays.asList(i,j)), length_by_path.get(Arrays.asList(i,k)) + length_by_path.get(Arrays.asList(k,j)));
                 length_by_path.put(Arrays.asList(i,j), update_length);
             }
         }
