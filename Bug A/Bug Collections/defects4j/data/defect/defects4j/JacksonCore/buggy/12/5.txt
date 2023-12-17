    protected boolean _isNextTokenNameMaybe(int i, String nameToMatch) throws IOException
    {
        // // // and this is back to standard nextToken()
        String name = (i == INT_QUOTE) ? _parseName() : _handleOddName(i);
        _parsingContext.setCurrentName(name);
        _currToken = JsonToken.FIELD_NAME;
        i = _skipColon();
        if (i == INT_QUOTE) {
            _tokenIncomplete = true;
            _nextToken = JsonToken.VALUE_STRING;
            return nameToMatch.equals(name);
        }
        // Ok: we must have a value... what is it?
        JsonToken t;
        switch (i) {
        case '-':
            t = _parseNegNumber();
            break;
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            t = _parsePosNumber(i);
            break;
        case 'f':
            _matchFalse();
            t = JsonToken.VALUE_FALSE;
            break;
        case 'n':
            _matchNull();
            t = JsonToken.VALUE_NULL;
            break;
        case 't':
            _matchTrue();
            t = JsonToken.VALUE_TRUE;
            break;
        case '[':
            t = JsonToken.START_ARRAY;
            break;
        case '{':
            t = JsonToken.START_OBJECT;
            break;
        default:
            t = _handleOddValue(i);
            break;
        }
        _nextToken = t;
        return nameToMatch.equals(name);
    }