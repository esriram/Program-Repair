--- 
+++ 
@@ -517,7 +517,7 @@
 		            _TIFFFieldWithTag(tif,TIFFTAG_STRIPBYTECOUNTS)->field_name);
 		if(EstimateStripByteCounts(tif, dir, dircount) < 0)
 		    goto bad;
-	} else if (td->td_nstrips > 1
+	} else if (td->td_nstrips > 2
 		   && td->td_compression == COMPRESSION_NONE
 		   && td->td_stripbytecount[0] != td->td_stripbytecount[1]) {
 		/*
