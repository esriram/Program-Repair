    public void writeNumber(short i) throws IOException {
        _append(JsonToken.VALUE_NUMBER_INT, Short.valueOf(i));
    }