--- 
+++ 
@@ -1,6 +1,6 @@
 public static ArrayList<ArrayList> subsequences(int a, int b, int k) {
     if (k == 0) {
-        return new ArrayList();
+        return new ArrayList<ArrayList>(Arrays.asList(new ArrayList()));
     }
 
     ArrayList ret = new ArrayList(50);
