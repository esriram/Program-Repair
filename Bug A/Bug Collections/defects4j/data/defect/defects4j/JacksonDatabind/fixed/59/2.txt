    public MapLikeType withContentValueHandler(Object h) {
        return new MapLikeType(_class, _bindings, _superClass,
                _superInterfaces, _keyType, _valueType.withValueHandler(h),
                _valueHandler, _typeHandler, _asStatic);
    }