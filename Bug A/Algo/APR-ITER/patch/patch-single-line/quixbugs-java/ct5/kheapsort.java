--- 
+++ 
@@ -5,7 +5,7 @@
     }
 
     ArrayList<Integer> output = new ArrayList<Integer>();
-    for (Integer x : arr) {
+for (Integer x : arr.subList(k,arr.size())) {//// if(!heap.isEmpty())////// } }// }//// if(!heap.isEmpty()) {// }
         heap.add(x);
         Integer popped = heap.poll();
         output.add(popped);
