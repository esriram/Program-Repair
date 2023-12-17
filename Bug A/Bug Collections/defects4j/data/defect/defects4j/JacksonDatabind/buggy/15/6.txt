    public boolean isEmpty(Object value)
    {
        Object delegateValue = convertValue(value);
        return _delegateSerializer.isEmpty(delegateValue);
    }