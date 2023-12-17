    public Object deserializeFromBoolean(JsonParser p, DeserializationContext ctxt) throws IOException
    {
        if (_delegateDeserializer != null) {
            if (!_valueInstantiator.canCreateFromBoolean()) {
                Object bean = _valueInstantiator.createUsingDelegate(ctxt,
                        _delegateDeserializer.deserialize(p, ctxt));
                if (_injectables != null) {
                    injectValues(ctxt, bean);
                }
                return bean;
            }
        }
        boolean value = (p.getCurrentToken() == JsonToken.VALUE_TRUE);
        return _valueInstantiator.createFromBoolean(ctxt, value);
    }