    public CSVParser(final Reader reader, final CSVFormat format, final long characterOffset, final long recordNumber)
            throws IOException {
        Assertions.notNull(reader, "reader");
        Assertions.notNull(format, "format");

        this.format = format;
        this.lexer = new Lexer(format, new ExtendedBufferedReader(reader));
        this.headerMap = this.initializeHeader();
        this.characterOffset = characterOffset;
        this.recordNumber = recordNumber - 1;
    }