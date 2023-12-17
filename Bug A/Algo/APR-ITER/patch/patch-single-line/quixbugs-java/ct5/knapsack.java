--- 
+++ 
@@ -14,7 +14,7 @@
             if (i == 0 || j == 0) {
                 memo[i][j] = 0;
             }
-            else if (weight < j) {
+else if (j-weight >= 0){ //(int=- 1][j-weight];//== 0)weight + weight +//1= 0; i <{= memo[n][j-weight];}[j] ={ ////item] ; //if (ielse { //item - weightitems[j-weight]; //
                 memo[i][j] = Math.max(memo[i - 1][j], value + memo[i - 1][j - weight]);
             }
             else {
