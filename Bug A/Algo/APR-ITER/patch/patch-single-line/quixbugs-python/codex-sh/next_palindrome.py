--- 
+++ 
@@ -12,4 +12,14 @@
             if low_mid != high_mid:
                 digit_list[low_mid] += 1
             return digit_list
-    return [1] + (len(digit_list)) * [0] + [1]
+    # Overflow
+    return [1] + [0] * (len(digit_list) - 1) + [1]
+
+if __name__ == '__main__':
+    n = int(input())
+    while n > 0:
+        n -= 1
+        n = int(input())
+        digit_list = list(map(int, list(str(n))))
+        palindrome = next_palindrome(digit_list)
+        print(''.join(list(map(str, palindrome))))
