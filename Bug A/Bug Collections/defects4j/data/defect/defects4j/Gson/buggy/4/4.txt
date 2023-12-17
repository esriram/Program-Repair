  public JsonWriter jsonValue(String value) throws IOException {
    if (value == null) {
      return nullValue();
    }
    writeDeferredName();
    beforeValue(false);
    out.append(value);
    return this;
  }