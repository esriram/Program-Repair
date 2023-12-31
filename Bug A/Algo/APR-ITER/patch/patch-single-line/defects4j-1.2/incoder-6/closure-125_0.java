--- 
+++ 
@@ -3,7 +3,7 @@
   JSType type = getJSType(constructor).restrictByNotNullOrUndefined();
   if (type.isConstructor() || type.isEmptyType() || type.isUnknownType()) {
     FunctionType fnType = type.toMaybeFunctionType();
-    if (fnType != null) {
+    if (fnType != null && fnType.hasInstanceType()) {
       visitParameterList(t, n, fnType);
       ensureTyped(t, n, fnType.getInstanceType());
     } else {
