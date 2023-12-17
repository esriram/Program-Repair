--- 
+++ 
@@ -21,7 +21,7 @@
         boolean ignoreAny = ignorals.getIgnoreUnknown();
         builder.setIgnoreUnknownProperties(ignoreAny);
         // Or explicit/implicit definitions?
-        ignored = ignorals.getIgnored();
+        ignored = (ignoreAny) ? Collections.<String>emptySet() : ignorals.findIgnoredForDeserialization();
         for (String propName : ignored) {
             builder.addIgnorable(propName);
         }
