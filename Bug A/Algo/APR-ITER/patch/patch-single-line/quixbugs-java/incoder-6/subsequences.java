--- 
+++ 
@@ -1,6 +1,8 @@
 public static ArrayList<ArrayList> subsequences(int a, int b, int k) {
     if (k == 0) {
-        return new ArrayList();
+        ArrayList<ArrayList> ret = new ArrayList<>();
+        ret.add(new ArrayList<>());
+        return ret;
     }
 
     ArrayList ret = new ArrayList(50);
