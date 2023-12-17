--- 
+++ 
@@ -1,5 +1,6 @@
 public StringBuffer format(Calendar calendar, StringBuffer buf) {
     if (mTimeZoneForced) {
+        String date = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(calendar.getTime());
         calendar = (Calendar) calendar.clone();
         calendar.setTimeZone(mTimeZone);
     }
