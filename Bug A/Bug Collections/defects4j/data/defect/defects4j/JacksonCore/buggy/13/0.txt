    public JsonGenerator enable(Feature f) {
        super.enable(f);
        if (f == Feature.QUOTE_FIELD_NAMES) {
            _cfgUnqNames = false;
        }
        return this;
    }