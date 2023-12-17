    public static final byte[] encodeQuotedPrintable(BitSet printable, byte[] bytes) {
        if (bytes == null) {
            return null;
        }
        if (printable == null) {
            printable = PRINTABLE_CHARS;
        }
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        // encode up to buffer.length - 3, the last three octets will be treated
        // separately for simplification of note #3
                // up to this length it is safe to add any byte, encoded or not
        for (byte c : bytes) {
            int b = c;
            if (b < 0) {
                b = 256 + b;
            }
            if (printable.get(b)) {
                buffer.write(b);
            } else {
                // rule #3: whitespace at the end of a line *must* be encoded

                // rule #5: soft line break
                encodeQuotedPrintable(b, buffer);
            }
        }

        // rule #3: whitespace at the end of a line *must* be encoded
        // if we would do a soft break line after this octet, encode whitespace

        // note #3: '=' *must not* be the ultimate or penultimate character
        // simplification: if < 6 bytes left, do a soft line break as we may need
        //                 exactly 6 bytes space for the last 2 bytes
            // rule #3: trailing whitespace shall be encoded

        return buffer.toByteArray();
    }