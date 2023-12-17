    protected JsonParserSequence(JsonParser[] parsers)
    {
        super(parsers[0]);
        _parsers = parsers;
        _nextParser = 1;
    }