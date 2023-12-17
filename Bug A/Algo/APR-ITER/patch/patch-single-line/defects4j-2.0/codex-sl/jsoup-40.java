--- 
+++ 
@@ -1,7 +1,6 @@
 public DocumentType(String name, String publicId, String systemId, String baseUri) {
     super(baseUri);
-
-    Validate.notEmpty(name);
+    // limited to html docs at this stage
     attr("name", name);
     attr("publicId", publicId);
     attr("systemId", systemId);
