    public CollectionLikeType withContentValueHandler(Object h) {
        return new CollectionLikeType(_class, _bindings,
                _superClass, _superInterfaces, _elementType.withValueHandler(h),
                _valueHandler, _typeHandler, _asStatic);
    }