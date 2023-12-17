--- 
+++ 
@@ -19,7 +19,7 @@
   TIFFGetField(in, TIFFTAG_IMAGELENGTH, &length);
   TIFFGetField(in, TIFFTAG_XRESOLUTION, &xres);
   TIFFGetField(in, TIFFTAG_YRESOLUTION, &yres);
-  TIFFGetFieldDefaulted(in, TIFFTAG_RESOLUTIONUNIT, &res_unit);
+  TIFFGetField(in, TIFFTAG_RESOLUTIONUNIT, &res_unit);
   scanlinesize = TIFFScanlineSize(in);
 
   image->bps = bps;
