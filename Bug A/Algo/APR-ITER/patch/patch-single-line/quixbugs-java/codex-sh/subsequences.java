--- 
+++ 
@@ -1,6 +1,8 @@
 public static ArrayList<ArrayList> subsequences(int a, int b, int k) {
     if (k == 0) {
-        return new ArrayList();
+        ArrayList<ArrayList> base = new ArrayList<ArrayList>(1);
+        base.add(new ArrayList());
+        return base;
     }
 
     ArrayList ret = new ArrayList(50);
