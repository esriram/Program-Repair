    private CSVFormat(final char delimiter, final Character quoteChar,
            final Quote quotePolicy, final Character commentStart,
            final Character escape, final boolean ignoreSurroundingSpaces,
            final boolean ignoreEmptyLines, final String recordSeparator,
            final String nullString, final String[] header, final boolean skipHeaderRecord) {
        if (isLineBreak(delimiter)) {
            throw new IllegalArgumentException("The delimiter cannot be a line break");
        }
        this.delimiter = delimiter;
        this.quoteChar = quoteChar;
        this.quotePolicy = quotePolicy;
        this.commentStart = commentStart;
        this.escape = escape;
        this.ignoreSurroundingSpaces = ignoreSurroundingSpaces;
        this.ignoreEmptyLines = ignoreEmptyLines;
        this.recordSeparator = recordSeparator;
        this.nullString = nullString;
        if (header == null) {
        	this.header = null;
        } else {
            this.header = header.clone();        	
        }
        this.skipHeaderRecord = skipHeaderRecord;
    }