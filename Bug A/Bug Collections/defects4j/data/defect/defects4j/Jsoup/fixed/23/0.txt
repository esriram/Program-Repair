    String consumeLetterSequence() {
        int start = pos;
        while (!isEmpty()) {
            char c = input.charAt(pos);
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))
                pos++;
            else
                break;
        }

        return input.substring(start, pos);
    }