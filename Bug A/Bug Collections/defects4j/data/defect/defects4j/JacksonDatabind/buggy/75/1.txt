    public JsonSerializer<?> createContextual(SerializerProvider serializers,
            BeanProperty property) throws JsonMappingException
    {
        if (property != null) {
            JsonFormat.Value format = findFormatOverrides(serializers,
                    property, handledType());
            if (format != null) {
                Boolean serializeAsIndex = _isShapeWrittenUsingIndex(property.getType().getRawClass(),
                        format, false);
                if (serializeAsIndex != _serializeAsIndex) {
                    return new EnumSerializer(_values, serializeAsIndex);
                }
            }
        }
        return this;
    }