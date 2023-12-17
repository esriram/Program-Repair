--- 
+++ 
@@ -1,5 +1,5 @@
 public boolean hasClass(String className) {
-    final String classAttr = attributes.get("class");
+    final String classAttr = this.attributes.getIgnoreCase("class");
     final int len = classAttr.length();
     final int wantLen = className.length();
 
