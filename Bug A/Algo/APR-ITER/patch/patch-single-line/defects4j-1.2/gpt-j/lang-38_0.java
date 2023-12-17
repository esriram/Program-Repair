--- 
+++ 
@@ -1,5 +1,6 @@
 public StringBuffer format(Calendar calendar, StringBuffer buf) {
     if (mTimeZoneForced) {
+        int startYear = calendar.get(Calendar.YEAR) - 1;
         calendar = (Calendar) calendar.clone();
         calendar.setTimeZone(mTimeZone);
     }
