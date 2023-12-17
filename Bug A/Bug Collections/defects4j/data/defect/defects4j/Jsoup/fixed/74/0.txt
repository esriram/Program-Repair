    public static boolean isActuallyWhitespace(int c){
        return c == ' ' || c == '\t' || c == '\n' || c == '\f' || c == '\r' || c == 160;
        // 160 is &nbsp; (non-breaking space). Not in the spec but expected.
    }