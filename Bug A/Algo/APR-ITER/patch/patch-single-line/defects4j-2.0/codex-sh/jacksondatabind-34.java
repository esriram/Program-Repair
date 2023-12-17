--- 
+++ 
@@ -5,7 +5,8 @@
     } else {
         Class<?> h = handledType();
         if (h == BigDecimal.class) {
-            visitFloatFormat(visitor, typeHint, JsonParser.NumberType.BIG_INTEGER);
+            // both BigDecimal and BigInteger have dedicated visits
+            visitFloatFormat(visitor, typeHint, JsonParser.NumberType.BIG_DECIMAL);
         } else {
             // otherwise bit unclear what to call... but let's try:
             /*JsonNumberFormatVisitor v2 =*/ visitor.expectNumberFormat(typeHint);
