--- 
+++ 
@@ -10,8 +10,8 @@
     } else if (methodsGuru.isCompareToMethod(invocation.getMethod())) {
         //see issue 184.
         //mocks by default should return 0 if references are the same, otherwise some other value because they are not the same. Hence we return 1 (anything but 0 is good).
-        //Only for compareTo() method by the Comparable interface
-        return 1;
+        //Only for compareTo() method by the Comparable interface.
+        return invocation.getMock() == invocation.getArguments()[0] ? 0 : 1;
     }
     
     Class<?> returnType = invocation.getMethod().getReturnType();
