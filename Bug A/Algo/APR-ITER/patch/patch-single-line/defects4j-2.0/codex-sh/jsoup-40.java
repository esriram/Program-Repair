--- 
+++ 
@@ -1,7 +1,5 @@
 public DocumentType(String name, String publicId, String systemId, String baseUri) {
     super(baseUri);
-
-    Validate.notEmpty(name);
     attr("name", name);
     attr("publicId", publicId);
     attr("systemId", systemId);
