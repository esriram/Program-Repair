    protected Object _weirdKey(DeserializationContext ctxt, String key, Exception e) throws IOException {
        return ctxt.handleWeirdKey(_keyClass, key, "problem: %s",
                e.getMessage());
    }