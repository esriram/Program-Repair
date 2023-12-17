    private JsonToken _nextBuffered(TokenFilterContext buffRoot) throws IOException
    {
        _exposedContext = buffRoot;
        TokenFilterContext ctxt = buffRoot;
        JsonToken t = ctxt.nextTokenToRead();
        if (t != null) {
            return t;
        }
        while (true) {
            // all done with buffered stuff?
            if (ctxt == _headContext) {
                throw _constructError("Internal error: failed to locate expected buffered tokens");
                /*
                _exposedContext = null;
                break;
                */
            }
            // If not, traverse down the context chain
            ctxt = _exposedContext.findChildOf(ctxt);
            _exposedContext = ctxt;
            if (ctxt == null) { // should never occur
                throw _constructError("Unexpected problem: chain of filtered context broken");
            }
            t = _exposedContext.nextTokenToRead();
            if (t != null) {
                return t;
            }
        }
    }