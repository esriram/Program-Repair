#include "Python.h"
#include "structmember.h"
#if PY_VERSION_HEX < 0x02060000 && !defined(Py_TYPE)
#define Py_TYPE(ob)     (((PyObject*)(ob))->ob_type)
#endif
#if PY_VERSION_HEX < 0x02050000 && !defined(PY_SSIZE_T_MIN)
typedef int Py_ssize_t;
#define PY_SSIZE_T_MAX INT_MAX
#define PY_SSIZE_T_MIN INT_MIN
#define PyInt_FromSsize_t PyInt_FromLong
#define PyInt_AsSsize_t PyInt_AsLong
#endif
#ifndef Py_IS_FINITE
#define Py_IS_FINITE(X) (!Py_IS_INFINITY(X) && !Py_IS_NAN(X))
#endif

#ifdef __GNUC__
#define UNUSED __attribute__((__unused__))
#else
#define UNUSED
#endif

#define PyScanner_Check(op) PyObject_TypeCheck(op, &PyScannerType)
#define PyScanner_CheckExact(op) (Py_TYPE(op) == &PyScannerType)
#define PyEncoder_Check(op) PyObject_TypeCheck(op, &PyEncoderType)
#define PyEncoder_CheckExact(op) (Py_TYPE(op) == &PyEncoderType)

static PyTypeObject PyScannerType;
static PyTypeObject PyEncoderType;

typedef struct _PyScannerObject {
    PyObject_HEAD
    PyObject *strict;
    PyObject *object_hook;
    PyObject *object_pairs_hook;
    PyObject *parse_float;
    PyObject *parse_int;
    PyObject *parse_constant;
    PyObject *memo;
} PyScannerObject;

static PyMemberDef scanner_members[] = {
    {"strict", T_OBJECT, offsetof(PyScannerObject, strict), READONLY, "strict"},
    {"object_hook", T_OBJECT, offsetof(PyScannerObject, object_hook), READONLY, "object_hook"},
    {"object_pairs_hook", T_OBJECT, offsetof(PyScannerObject, object_pairs_hook), READONLY},
    {"parse_float", T_OBJECT, offsetof(PyScannerObject, parse_float), READONLY, "parse_float"},
    {"parse_int", T_OBJECT, offsetof(PyScannerObject, parse_int), READONLY, "parse_int"},
    {"parse_constant", T_OBJECT, offsetof(PyScannerObject, parse_constant), READONLY, "parse_constant"},
    {NULL}
};

typedef struct _PyEncoderObject {
    PyObject_HEAD
    PyObject *markers;
    PyObject *defaultfn;
    PyObject *encoder;
    PyObject *indent;
    PyObject *key_separator;
    PyObject *item_separator;
    PyObject *sort_keys;
    PyObject *skipkeys;
    int fast_encode;
    int allow_nan;
} PyEncoderObject;

static PyMemberDef encoder_members[] = {
    {"markers", T_OBJECT, offsetof(PyEncoderObject, markers), READONLY, "markers"},
    {"default", T_OBJECT, offsetof(PyEncoderObject, defaultfn), READONLY, "default"},
    {"encoder", T_OBJECT, offsetof(PyEncoderObject, encoder), READONLY, "encoder"},
    {"indent", T_OBJECT, offsetof(PyEncoderObject, indent), READONLY, "indent"},
    {"key_separator", T_OBJECT, offsetof(PyEncoderObject, key_separator), READONLY, "key_separator"},
    {"item_separator", T_OBJECT, offsetof(PyEncoderObject, item_separator), READONLY, "item_separator"},
    {"sort_keys", T_OBJECT, offsetof(PyEncoderObject, sort_keys), READONLY, "sort_keys"},
    {"skipkeys", T_OBJECT, offsetof(PyEncoderObject, skipkeys), READONLY, "skipkeys"},
    {NULL}
};

static PyObject *
ascii_escape_unicode(PyObject *pystr);
static PyObject *
py_encode_basestring_ascii(PyObject* self UNUSED, PyObject *pystr);
void init_json(void);
static PyObject *
scan_once_unicode(PyScannerObject *s, PyObject *pystr, Py_ssize_t idx, Py_ssize_t *next_idx_ptr);
static PyObject *
_build_rval_index_tuple(PyObject *rval, Py_ssize_t idx);
static PyObject *
scanner_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static int
scanner_init(PyObject *self, PyObject *args, PyObject *kwds);
static void
scanner_dealloc(PyObject *self);
static int
scanner_clear(PyObject *self);
static PyObject *
encoder_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
static int
encoder_init(PyObject *self, PyObject *args, PyObject *kwds);
static void
encoder_dealloc(PyObject *self);
static int
encoder_clear(PyObject *self);
static int
encoder_listencode_list(PyEncoderObject *s, PyObject *rval, PyObject *seq, Py_ssize_t indent_level);
static int
encoder_listencode_obj(PyEncoderObject *s, PyObject *rval, PyObject *obj, Py_ssize_t indent_level);
static int
encoder_listencode_dict(PyEncoderObject *s, PyObject *rval, PyObject *dct, Py_ssize_t indent_level);
static PyObject *
_encoded_const(PyObject *obj);
static void
raise_errmsg(char *msg, PyObject *s, Py_ssize_t end);
static PyObject *
encoder_encode_string(PyEncoderObject *s, PyObject *obj);
static int
_convertPyInt_AsSsize_t(PyObject *o, Py_ssize_t *size_ptr);
static PyObject *
_convertPyInt_FromSsize_t(Py_ssize_t *size_ptr);
static PyObject *
encoder_encode_float(PyEncoderObject *s, PyObject *obj);

#define S_CHAR(c) (c >= ' ' && c <= '~' && c != '\\' && c != '"')
#define IS_WHITESPACE(c) (((c) == ' ') || ((c) == '\t') || ((c) == '\n') || ((c) == '\r'))

#define MIN_EXPANSION 6
#ifdef Py_UNICODE_WIDE
#define MAX_EXPANSION (2 * MIN_EXPANSION)
#else
#define MAX_EXPANSION MIN_EXPANSION
#endif

static int
_convertPyInt_AsSsize_t(PyObject *o, Py_ssize_t *size_ptr)
{
    /* PyObject to Py_ssize_t converter */
    *size_ptr = PyLong_AsSsize_t(o);
    if (*size_ptr == -1 && PyErr_Occurred())
        return 0;
    return 1;
}

static PyObject *
_convertPyInt_FromSsize_t(Py_ssize_t *size_ptr)
{
    /* Py_ssize_t to PyObject converter */
    return PyLong_FromSsize_t(*size_ptr);
}

static Py_ssize_t
ascii_escape_unichar(Py_UNICODE c, Py_UNICODE *output, Py_ssize_t chars)
{
    /* Escape unicode code point c to ASCII escape sequences
    in char *output. output must have at least 12 bytes unused to
    accommodate an escaped surrogate pair "\uXXXX\uXXXX" */
    output[chars++] = '\\';
    switch (c) {
        case '\\': output[chars++] = c; break;
        case '"': output[chars++] = c; break;
        case '\b': output[chars++] = 'b'; break;
        case '\f': output[chars++] = 'f'; break;
        case '\n': output[chars++] = 'n'; break;
        case '\r': output[chars++] = 'r'; break;
        case '\t': output[chars++] = 't'; break;
        default:
#ifdef Py_UNICODE_WIDE
            if (c >= 0x10000) {
                /* UTF-16 surrogate pair */
                Py_UNICODE v = c - 0x10000;
                c = 0xd800 | ((v >> 10) & 0x3ff);
                output[chars++] = 'u';
                output[chars++] = "0123456789abcdef"[(c >> 12) & 0xf];
                output[chars++] = "0123456789abcdef"[(c >>  8) & 0xf];
                output[chars++] = "0123456789abcdef"[(c >>  4) & 0xf];
                output[chars++] = "0123456789abcdef"[(c      ) & 0xf];
                c = 0xdc00 | (v & 0x3ff);
                output[chars++] = '\\';
            }
#endif
            output[chars++] = 'u';
            output[chars++] = "0123456789abcdef"[(c >> 12) & 0xf];
            output[chars++] = "0123456789abcdef"[(c >>  8) & 0xf];
            output[chars++] = "0123456789abcdef"[(c >>  4) & 0xf];
            output[chars++] = "0123456789abcdef"[(c      ) & 0xf];
    }
    return chars;
}

static PyObject *
ascii_escape_unicode(PyObject *pystr)
{
    /* Take a PyUnicode pystr and return a new ASCII-only escaped PyUnicode */
    Py_ssize_t i;
    Py_ssize_t input_chars;
    Py_ssize_t output_size;
    Py_ssize_t max_output_size;
    Py_ssize_t chars;
    PyObject *rval;
    Py_UNICODE *output;
    Py_UNICODE *input_unicode;

    input_chars = PyUnicode_GET_SIZE(pystr);
    input_unicode = PyUnicode_AS_UNICODE(pystr);

    /* One char input can be up to 6 chars output, estimate 4 of these */
    output_size = 2 + (MIN_EXPANSION * 4) + input_chars;
    max_output_size = 2 + (input_chars * MAX_EXPANSION);
    rval = PyUnicode_FromStringAndSize(NULL, output_size);
    if (rval == NULL) {
        return NULL;
    }
    output = PyUnicode_AS_UNICODE(rval);
    chars = 0;
    output[chars++] = '"';
    for (i = 0; i < input_chars; i++) {
        Py_UNICODE c = input_unicode[i];
        if (S_CHAR(c)) {
            output[chars++] = c;
        }
        else {
            chars = ascii_escape_unichar(c, output, chars);
        }
        if (output_size - chars < (1 + MAX_EXPANSION)) {
            /* There's more than four, so let's resize by a lot */
            Py_ssize_t new_output_size = output_size * 2;
            /* This is an upper bound */
            if (new_output_size > max_output_size) {
                new_output_size = max_output_size;
            }
            /* Make sure that the output size changed before resizing */
            if (new_output_size != output_size) {
                output_size = new_output_size;
                if (PyUnicode_Resize(&rval, output_size) == -1) {
                    return NULL;
                }
                output = PyUnicode_AS_UNICODE(rval);
            }
        }
    }
    output[chars++] = '"';
    if (PyUnicode_Resize(&rval, chars) == -1) {
        return NULL;
    }
    return rval;
}

static void
raise_errmsg(char *msg, PyObject *s, Py_ssize_t end)
{
    /* Use the Python function json.decoder.errmsg to raise a nice
    looking ValueError exception */
    static PyObject *errmsg_fn = NULL;
    PyObject *pymsg;
    if (errmsg_fn == NULL) {
        PyObject *decoder = PyImport_ImportModule("json.decoder");
        if (decoder == NULL)
            return;
        errmsg_fn = PyObject_GetAttrString(decoder, "errmsg");
        Py_DECREF(decoder);
        if (errmsg_fn == NULL)
            return;
    }
    pymsg = PyObject_CallFunction(errmsg_fn, "(zOO&)", msg, s, _convertPyInt_FromSsize_t, &end);
    if (pymsg) {
        PyErr_SetObject(PyExc_ValueError, pymsg);
        Py_DECREF(pymsg);
    }
}

static PyObject *
join_list_unicode(PyObject *lst)
{
    /* return u''.join(lst) */
    static PyObject *sep = NULL;
    if (sep == NULL) {
        sep = PyUnicode_FromStringAndSize("", 0);
        if (sep == NULL)
            return NULL;
    }
    return PyUnicode_Join(sep, lst);
}

static PyObject *
_build_rval_index_tuple(PyObject *rval, Py_ssize_t idx) {
    /* return (rval, idx) tuple, stealing reference to rval */
    PyObject *tpl;
    PyObject *pyidx;
    /*
    steal a reference to rval, returns (rval, idx)
    */
    if (rval == NULL) {
        return NULL;
    }
    pyidx = PyLong_FromSsize_t(idx);
    if (pyidx == NULL) {
        Py_DECREF(rval);
        return NULL;
    }
    tpl = PyTuple_New(2);
    if (tpl == NULL) {
        Py_DECREF(pyidx);
        Py_DECREF(rval);
        return NULL;
    }
    PyTuple_SET_ITEM(tpl, 0, rval);
    PyTuple_SET_ITEM(tpl, 1, pyidx);
    return tpl;
}

#define APPEND_OLD_CHUNK \
    if (chunk != NULL) { \
        if (chunks == NULL) { \
            chunks = PyList_New(0); \
            if (chunks == NULL) { \
                goto bail; \
            } \
        } \
        if (PyList_Append(chunks, chunk)) { \
            Py_DECREF(chunk); \
            goto bail; \
        } \
        Py_CLEAR(chunk); \
    }

static PyObject *
scanstring_unicode(PyObject *pystr, Py_ssize_t end, int strict, Py_ssize_t *next_end_ptr)
{
    /* Read the JSON string from PyUnicode pystr.
    end is the index of the first character after the quote.
    if strict is zero then literal control characters are allowed
    *next_end_ptr is a return-by-reference index of the character
        after the end quote

    Return value is a new PyUnicode
    */
    PyObject *rval = NULL;
    Py_ssize_t len = PyUnicode_GET_SIZE(pystr);
    Py_ssize_t begin = end - 1;
    Py_ssize_t next /* = begin */;
    const Py_UNICODE *buf = PyUnicode_AS_UNICODE(pystr);
    PyObject *chunks = NULL;
    PyObject *chunk = NULL;

    if (end < 0 || len <= end) {
        PyErr_SetString(PyExc_ValueError, "end is out of bounds");
        goto bail;
    }
    while (1) {
        /* Find the end of the string or the next escape */
        Py_UNICODE c = 0;
        for (next = end; next < len; next++) {
            c = buf[next];
            if (c == '"' || c == '\\') {
                break;
            }
            else if (strict && c <= 0x1f) {
                raise_errmsg("Invalid control character at", pystr, next);
                goto bail;
            }
        }
        if (!(c == '"' || c == '\\')) {
            raise_errmsg("Unterminated string starting at", pystr, begin);
            goto bail;
        }
        /* Pick up this chunk if it's not zero length */
        if (next != end) {
            APPEND_OLD_CHUNK
            chunk = PyUnicode_FromUnicode(&buf[end], next - end);
            if (chunk == NULL) {
                goto bail;
            }
        }
        next++;
        if (c == '"') {
            end = next;
            break;
        }
        if (next == len) {
            raise_errmsg("Unterminated string starting at", pystr, begin);
            goto bail;
        }
        c = buf[next];
        if (c != 'u') {
            /* Non-unicode backslash escapes */
            end = next + 1;
            switch (c) {
                case '"': break;
                case '\\': break;
                case '/': break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case 'n': c = '\n'; break;
                case 'r': c = '\r'; break;
                case 't': c = '\t'; break;
                default: c = 0;
            }
            if (c == 0) {
                raise_errmsg("Invalid \\escape", pystr, end - 2);
                goto bail;
            }
        }
        else {
            c = 0;
            next++;
            end = next + 4;
            if (end >= len) {
                raise_errmsg("Invalid \\uXXXX escape", pystr, next - 1);
                goto bail;
            }
            /* Decode 4 hex digits */
            for (; next < end; next++) {
                Py_UNICODE digit = buf[next];
                c <<= 4;
                switch (digit) {
                    case '0': case '1': case '2': case '3': case '4':
                    case '5': case '6': case '7': case '8': case '9':
                        c |= (digit - '0'); break;
                    case 'a': case 'b': case 'c': case 'd': case 'e':
                    case 'f':
                        c |= (digit - 'a' + 10); break;
                    case 'A': case 'B': case 'C': case 'D': case 'E':
                    case 'F':
                        c |= (digit - 'A' + 10); break;
                    default:
                        raise_errmsg("Invalid \\uXXXX escape", pystr, end - 5);
                        goto bail;
                }
            }
#ifdef Py_UNICODE_WIDE
            /* Surrogate pair */
            if ((c & 0xfc00) == 0xd800) {
                Py_UNICODE c2 = 0;
                if (end + 6 >= len) {
                    raise_errmsg("Unpaired high surrogate", pystr, end - 5);
                    goto bail;
                }
                if (buf[next++] != '\\' || buf[next++] != 'u') {
                    raise_errmsg("Unpaired high surrogate", pystr, end - 5);
                    goto bail;
                }
                end += 6;
                /* Decode 4 hex digits */
                for (; next < end; next++) {
                    Py_UNICODE digit = buf[next];
                    c2 <<= 4;
                    switch (digit) {
                        case '0': case '1': case '2': case '3': case '4':
                        case '5': case '6': case '7': case '8': case '9':
                            c2 |= (digit - '0'); break;
                        case 'a': case 'b': case 'c': case 'd': case 'e':
                        case 'f':
                            c2 |= (digit - 'a' + 10); break;
                        case 'A': case 'B': case 'C': case 'D': case 'E':
                        case 'F':
                            c2 |= (digit - 'A' + 10); break;
                        default:
                            raise_errmsg("Invalid \\uXXXX escape", pystr, end - 5);
                            goto bail;
                    }
                }
                if ((c2 & 0xfc00) != 0xdc00) {
                    raise_errmsg("Unpaired high surrogate", pystr, end - 5);
                    goto bail;
                }
                c = 0x10000 + (((c - 0xd800) << 10) | (c2 - 0xdc00));
            }
            else if ((c & 0xfc00) == 0xdc00) {
                raise_errmsg("Unpaired low surrogate", pystr, end - 5);
                goto bail;
            }
#endif
        }
        APPEND_OLD_CHUNK
        chunk = PyUnicode_FromUnicode(&c, 1);
        if (chunk == NULL) {
            goto bail;
        }
    }

    if (chunks == NULL) {
        if (chunk != NULL)
            rval = chunk;
        else
            rval = PyUnicode_FromStringAndSize("", 0);
    }
    else {
        APPEND_OLD_CHUNK
        rval = join_list_unicode(chunks);
        if (rval == NULL) {
            goto bail;
        }
        Py_CLEAR(chunks);
    }

    *next_end_ptr = end;
    return rval;
bail:
    *next_end_ptr = -1;
    Py_XDECREF(chunks);
    Py_XDECREF(chunk);
    return NULL;
}

PyDoc_STRVAR(pydoc_scanstring,
    "scanstring(string, end, strict=True) -> (string, end)\n"
    "\n"
    "Scan the string s for a JSON string. End is the index of the\n"
    "character in s after the quote that started the JSON string.\n"
    "Unescapes all valid JSON string escape sequences and raises ValueError\n"
    "on attempt to decode an invalid string. If strict is False then literal\n"
    "control characters are allowed in the string.\n"
    "\n"
    "Returns a tuple of the decoded string and the index of the character in s\n"
    "after the end quote."
);

static PyObject *
py_scanstring(PyObject* self UNUSED, PyObject *args)
{
    PyObject *pystr;
    PyObject *rval;
    Py_ssize_t end;
    Py_ssize_t next_end = -1;
    int strict = 1;
    if (!PyArg_ParseTuple(args, "OO&|i:scanstring", &pystr, _convertPyInt_AsSsize_t, &end, &strict)) {
        return NULL;
    }
    if (PyUnicode_Check(pystr)) {
        rval = scanstring_unicode(pystr, end, strict, &next_end);
    }
    else {
        PyErr_Format(PyExc_TypeError,
                     "first argument must be a string, not %.80s",
                     Py_TYPE(pystr)->tp_name);
        return NULL;
    }
    return _build_rval_index_tuple(rval, next_end);
}

PyDoc_STRVAR(pydoc_encode_basestring_ascii,
    "encode_basestring_ascii(string) -> string\n"
    "\n"
    "Return an ASCII-only JSON representation of a Python string"
);

static PyObject *
py_encode_basestring_ascii(PyObject* self UNUSED, PyObject *pystr)
{
    PyObject *rval;
    /* Return an ASCII-only JSON representation of a Python string */
    /* METH_O */
    if (PyUnicode_Check(pystr)) {
        rval = ascii_escape_unicode(pystr);
    }
    else {
        PyErr_Format(PyExc_TypeError,
                     "first argument must be a string, not %.80s",
                     Py_TYPE(pystr)->tp_name);
        return NULL;
    }
    return rval;
}

static void
scanner_dealloc(PyObject *self)
{
    /* Deallocate scanner object */
    scanner_clear(self);
    Py_TYPE(self)->tp_free(self);
}

static int
scanner_traverse(PyObject *self, visitproc visit, void *arg)
{
    PyScannerObject *s;
    assert(PyScanner_Check(self));
    s = (PyScannerObject *)self;
    Py_VISIT(s->strict);
    Py_VISIT(s->object_hook);
    Py_VISIT(s->object_pairs_hook);
    Py_VISIT(s->parse_float);
    Py_VISIT(s->parse_int);
    Py_VISIT(s->parse_constant);
    return 0;
}

static int
scanner_clear(PyObject *self)
{
    PyScannerObject *s;
    assert(PyScanner_Check(self));
    s = (PyScannerObject *)self;
    Py_CLEAR(s->strict);
    Py_CLEAR(s->object_hook);
    Py_CLEAR(s->object_pairs_hook);
    Py_CLEAR(s->parse_float);
    Py_CLEAR(s->parse_int);
    Py_CLEAR(s->parse_constant);
    Py_CLEAR(s->memo);
    return 0;
}

static PyObject *
_parse_object_unicode(PyScannerObject *s, PyObject *pystr, Py_ssize_t idx, Py_ssize_t *next_idx_ptr) {
    /* Read a JSON object from PyUnicode pystr.
    idx is the index of the first character after the opening curly brace.
    *next_idx_ptr is a return-by-reference index to the first character after
        the closing curly brace.

    Returns a new PyObject (usually a dict, but object_hook can change that)
    */
    Py_UNICODE *str = PyUnicode_AS_UNICODE(pystr);
    Py_ssize_t end_idx = PyUnicode_GET_SIZE(pystr) - 1;
    PyObject *val = NULL;
    PyObject *rval = NULL;
    PyObject *key = NULL;
    int strict = PyObject_IsTrue(s->strict);
    int has_pairs_hook = (s->object_pairs_hook != Py_None);
    Py_ssize_t next_idx;

    if (has_pairs_hook)
        rval = PyList_New(0);
    else
        rval = PyDict_New();
    if (rval == NULL)
        return NULL;

    /* skip whitespace after { */
    while (idx <= end_idx && IS_WHITESPACE(str[idx])) idx++;

    /* only loop if the object is non-empty */
    if (idx <= end_idx && str[idx] != '}') {
        while (idx <= end_idx) {
            PyObject *memokey;

            /* read key */
            if (str[idx] != '"') {
                raise_errmsg("Expecting property name", pystr, idx);
                goto bail;
            }
            key = scanstring_unicode(pystr, idx + 1, strict, &next_idx);
            if (key == NULL)
                goto bail;
            memokey = PyDict_GetItem(s->memo, key);
            if (memokey != NULL) {
                Py_INCREF(memokey);
                Py_DECREF(key);
                key = memokey;
            }
            else {
                if (PyDict_SetItem(s->memo, key, key) < 0)
                    goto bail;
            }
            idx = next_idx;

            /* skip whitespace between key and : delimiter, read :, skip whitespace */
            while (idx <= end_idx && IS_WHITESPACE(str[idx])) idx++;
            if (idx > end_idx || str[idx] != ':') {
                raise_errmsg("Expecting : delimiter", pystr, idx);
                goto bail;
            }
            idx++;
            while (idx <= end_idx && IS_WHITESPACE(str[idx])) idx++;

            /* read any JSON term */
            val = scan_once_unicode(s, pystr, idx, &next_idx);
            if (val == NULL)
                goto bail;

            if (has_pairs_hook) {
                PyObject *item = PyTuple_Pack(2, key, val);
                if (item == NULL)
                    goto bail;
                Py_CLEAR(key);
                Py_CLEAR(val);
                if (PyList_Append(rval, item) == -1) {
                    Py_DECREF(item);
                    goto bail;
                }
                Py_DECREF(item);
            }
            else {
                if (PyDict_SetItem(rval, key, val) < 0)
                    goto bail;
                Py_CLEAR(key);
                Py_CLEAR(val);
            }
            idx = next_idx;

            /* skip whitespace before } or , */
            while (idx <= end_idx && IS_WHITESPACE(str[idx])) idx++;

            /* bail if the object is closed or we didn't get the , delimiter */
            if (idx > end_idx) break;
            if (str[idx] == '}') {
                break;
            }
            else if (str[idx] != ',') {
                raise_errmsg("Expecting , delimiter", pystr, idx);
                goto bail;
            }
            idx++;

            /* skip whitespace after , delimiter */
            while (idx <= end_idx && IS_WHITESPACE(str[idx])) idx++;
        }
    }

    /* verify that idx < end_idx, str[idx] should be '}' */
    if (idx > end_idx || str[idx] != '}') {
        raise_errmsg("Expecting object", pystr, end_idx);
        goto bail;
    }

    *next_idx_ptr = idx + 1;

    if (has_pairs_hook) {
        val = PyObject_CallFunctionObjArgs(s->object_pairs_hook, rval, NULL);
        Py_DECREF(rval);
        return val;
    }

    /* if object_hook is not None: rval = object_hook(rval) */
    if (s->object_hook != Py_None) {
        val = PyObject_CallFunctionObjArgs(s->object_hook, rval, NULL);
        Py_DECREF(rval);
        return val;
    }
    return rval;
bail:
    Py_XDECREF(key);
    Py_XDECREF(val);
    Py_XDECREF(rval);
    return NULL;
}

static PyObject *
_parse_array_unicode(PyScannerObject *s, PyObject *pystr, Py_ssize_t idx, Py_ssize_t *next_idx_ptr) {
    /* Read a JSON array from PyString pystr.
    idx is the index of the first character after the opening brace.
    *next_idx_ptr is a return-by-reference index to the first character after
        the closing brace.

    Returns a new PyList
    */
    Py_UNICODE *str = PyUnicode_AS_UNICODE(pystr);
    Py_ssize_t end_idx = PyUnicode_GET_SIZE(pystr) - 1;
    PyObject *val = NULL;
    PyObject *rval = PyList_New(0);
    Py_ssize_t next_idx;
    if (rval == NULL)
        return NULL;

    /* skip whitespace after [ */
    while (idx <= end_idx && IS_WHITESPACE(str[idx])) idx++;

    /* only loop if the array is non-empty */
    if (idx <= end_idx && str[idx] != ']') {
        while (idx <= end_idx) {

            /* read any JSON term  */
            val = scan_once_unicode(s, pystr, idx, &next_idx);
            if (val == NULL)
                goto bail;

            if (PyList_Append(rval, val) == -1)
                goto bail;

            Py_CLEAR(val);
            idx = next_idx;

            /* skip whitespace between term and , */
            while (idx <= end_idx && IS_WHITESPACE(str[idx])) idx++;

            /* bail if the array is closed or we didn't get the , delimiter */
            if (idx > end_idx) break;
            if (str[idx] == ']') {
                break;
            }
            else if (str[idx] != ',') {
                raise_errmsg("Expecting , delimiter", pystr, idx);
                goto bail;
            }
            idx++;

            /* skip whitespace after , */
            while (idx <= end_idx && IS_WHITESPACE(str[idx])) idx++;
        }
    }

    /* verify that idx < end_idx, str[idx] should be ']' */
    if (idx > end_idx || str[idx] != ']') {
        raise_errmsg("Expecting object", pystr, end_idx);
        goto bail;
    }
    *next_idx_ptr = idx + 1;
    return rval;
bail:
    Py_XDECREF(val);
    Py_DECREF(rval);
    return NULL;
}

static PyObject *
_parse_constant(PyScannerObject *s, char *constant, Py_ssize_t idx, Py_ssize_t *next_idx_ptr) {
    /* Read a JSON constant from PyString pystr.
    constant is the constant string that was found
        ("NaN", "Infinity", "-Infinity").
    idx is the index of the first character of the constant
    *next_idx_ptr is a return-by-reference index to the first character after
        the constant.

    Returns the result of parse_constant
    */
    PyObject *cstr;
    PyObject *rval;
    /* constant is "NaN", "Infinity", or "-Infinity" */
    cstr = PyUnicode_InternFromString(constant);
    if (cstr == NULL)
        return NULL;

    /* rval = parse_constant(constant) */
    rval = PyObject_CallFunctionObjArgs(s->parse_constant, cstr, NULL);
    idx += PyUnicode_GET_SIZE(cstr);
    Py_DECREF(cstr);
    *next_idx_ptr = idx;
    return rval;
}

static PyObject *
_match_number_unicode(PyScannerObject *s, PyObject *pystr, Py_ssize_t start, Py_ssize_t *next_idx_ptr) {
    /* Read a JSON number from PyUnicode pystr.
    idx is the index of the first character of the number
    *next_idx_ptr is a return-by-reference index to the first character after
        the number.

    Returns a new PyObject representation of that number:
        PyInt, PyLong, or PyFloat.
        May return other types if parse_int or parse_float are set
    */
    Py_UNICODE *str = PyUnicode_AS_UNICODE(pystr);
    Py_ssize_t end_idx = PyUnicode_GET_SIZE(pystr) - 1;
    Py_ssize_t idx = start;
    int is_float = 0;
    PyObject *rval;
    PyObject *numstr = NULL;
    PyObject *custom_func;

    /* read a sign if it's there, make sure it's not the end of the string */
    if (str[idx] == '-') {
        idx++;
        if (idx > end_idx) {
            PyErr_SetNone(PyExc_StopIteration);
            return NULL;
        }
    }

    /* read as many integer digits as we find as long as it doesn't start with 0 */
    if (str[idx] >= '1' && str[idx] <= '9') {
        idx++;
        while (idx <= end_idx && str[idx] >= '0' && str[idx] <= '9') idx++;
    }
    /* if it starts with 0 we only expect one integer digit */
    else if (str[idx] == '0') {
        idx++;
    }
    /* no integer digits, error */
    else {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }

    /* if the next char is '.' followed by a digit then read all float digits */
    if (idx < end_idx && str[idx] == '.' && str[idx + 1] >= '0' && str[idx + 1] <= '9') {
        is_float = 1;
        idx += 2;
        while (idx <= end_idx && str[idx] >= '0' && str[idx] <= '9') idx++;
    }

    /* if the next char is 'e' or 'E' then maybe read the exponent (or backtrack) */
    if (idx < end_idx && (str[idx] == 'e' || str[idx] == 'E')) {
        Py_ssize_t e_start = idx;
        idx++;

        /* read an exponent sign if present */
        if (idx < end_idx && (str[idx] == '-' || str[idx] == '+')) idx++;

        /* read all digits */
        while (idx <= end_idx && str[idx] >= '0' && str[idx] <= '9') idx++;

        /* if we got a digit, then parse as float. if not, backtrack */
        if (str[idx - 1] >= '0' && str[idx - 1] <= '9') {
            is_float = 1;
        }
        else {
            idx = e_start;
        }
    }

    if (is_float && s->parse_float != (PyObject *)&PyFloat_Type)
        custom_func = s->parse_float;
    else if (!is_float && s->parse_int != (PyObject *) &PyLong_Type)
        custom_func = s->parse_int;
    else
        custom_func = NULL;

    if (custom_func) {
        /* copy the section we determined to be a number */
        numstr = PyUnicode_FromUnicode(&str[start], idx - start);
        if (numstr == NULL)
            return NULL;
        rval = PyObject_CallFunctionObjArgs(custom_func, numstr, NULL);
    }
    else {
        Py_ssize_t i, n;
        char *buf;
        /* Straight conversion to ASCII, to avoid costly conversion of
           decimal unicode digits (which cannot appear here) */
        n = idx - start;
        numstr = PyBytes_FromStringAndSize(NULL, n);
        if (numstr == NULL)
            return NULL;
        buf = PyBytes_AS_STRING(numstr);
        for (i = 0; i < n; i++) {
            buf[i] = (char) str[i + start];
        }
        if (is_float)
            rval = PyFloat_FromString(numstr);
        else
            rval = PyLong_FromString(buf, NULL, 10);
    }
    Py_DECREF(numstr);
    *next_idx_ptr = idx;
    return rval;
}

static PyObject *
scan_once_unicode(PyScannerObject *s, PyObject *pystr, Py_ssize_t idx, Py_ssize_t *next_idx_ptr)
{
    /* Read one JSON term (of any kind) from PyUnicode pystr.
    idx is the index of the first character of the term
    *next_idx_ptr is a return-by-reference index to the first character after
        the number.

    Returns a new PyObject representation of the term.
    */
    PyObject *res;
    Py_UNICODE *str = PyUnicode_AS_UNICODE(pystr);
    Py_ssize_t length = PyUnicode_GET_SIZE(pystr);
    if (idx >= length) {
        PyErr_SetNone(PyExc_StopIteration);
        return NULL;
    }
    switch (str[idx]) {
        case '"':
            /* string */
            return scanstring_unicode(pystr, idx + 1,
                PyObject_IsTrue(s->strict),
                next_idx_ptr);
        case '{':
            /* object */
            if (Py_EnterRecursiveCall(" while decoding a JSON object "
                                      "from a unicode string"))
                return NULL;
            res = _parse_object_unicode(s, pystr, idx + 1, next_idx_ptr);
            Py_LeaveRecursiveCall();
            return res;
        case '[':
            /* array */
            if (Py_EnterRecursiveCall(" while decoding a JSON array "
                                      "from a unicode string"))
                return NULL;
            res = _parse_array_unicode(s, pystr, idx + 1, next_idx_ptr);
            Py_LeaveRecursiveCall();
            return res;
        case 'n':
            /* null */
            if ((idx + 3 < length) && str[idx + 1] == 'u' && str[idx + 2] == 'l' && str[idx + 3] == 'l') {
                Py_INCREF(Py_None);
                *next_idx_ptr = idx + 4;
                return Py_None;
            }
            break;
        case 't':
            /* true */
            if ((idx + 3 < length) && str[idx + 1] == 'r' && str[idx + 2] == 'u' && str[idx + 3] == 'e') {
                Py_INCREF(Py_True);
                *next_idx_ptr = idx + 4;
                return Py_True;
            }
            break;
        case 'f':
            /* false */
            if ((idx + 4 < length) && str[idx + 1] == 'a' && str[idx + 2] == 'l' && str[idx + 3] == 's' && str[idx + 4] == 'e') {
                Py_INCREF(Py_False);
                *next_idx_ptr = idx + 5;
                return Py_False;
            }
            break;
        case 'N':
            /* NaN */
            if ((idx + 2 < length) && str[idx + 1] == 'a' && str[idx + 2] == 'N') {
                return _parse_constant(s, "NaN", idx, next_idx_ptr);
            }
            break;
        case 'I':
            /* Infinity */
            if ((idx + 7 < length) && str[idx + 1] == 'n' && str[idx + 2] == 'f' && str[idx + 3] == 'i' && str[idx + 4] == 'n' && str[idx + 5] == 'i' && str[idx + 6] == 't' && str[idx + 7] == 'y') {
                return _parse_constant(s, "Infinity", idx, next_idx_ptr);
            }
            break;
        case '-':
            /* -Infinity */
            if ((idx + 8 < length) && str[idx + 1] == 'I' && str[idx + 2] == 'n' && str[idx + 3] == 'f' && str[idx + 4] == 'i' && str[idx + 5] == 'n' && str[idx + 6] == 'i' && str[idx + 7] == 't' && str[idx + 8] == 'y') {
                return _parse_constant(s, "-Infinity", idx, next_idx_ptr);
            }
            break;
    }
    /* Didn't find a string, object, array, or named constant. Look for a number. */
    return _match_number_unicode(s, pystr, idx, next_idx_ptr);
}

static PyObject *
scanner_call(PyObject *self, PyObject *args, PyObject *kwds)
{
    /* Python callable interface to scan_once_{str,unicode} */
    PyObject *pystr;
    PyObject *rval;
    Py_ssize_t idx;
    Py_ssize_t next_idx = -1;
    static char *kwlist[] = {"string", "idx", NULL};
    PyScannerObject *s;
    assert(PyScanner_Check(self));
    s = (PyScannerObject *)self;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO&:scan_once", kwlist, &pystr, _convertPyInt_AsSsize_t, &idx))
        return NULL;

    if (PyUnicode_Check(pystr)) {
        rval = scan_once_unicode(s, pystr, idx, &next_idx);
    }
    else {
        PyErr_Format(PyExc_TypeError,
                 "first argument must be a string, not %.80s",
                 Py_TYPE(pystr)->tp_name);
        return NULL;
    }
    PyDict_Clear(s->memo);
    if (rval == NULL)
        return NULL;
    return _build_rval_index_tuple(rval, next_idx);
}

static PyObject *
scanner_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyScannerObject *s;
    s = (PyScannerObject *)type->tp_alloc(type, 0);
    if (s != NULL) {
        s->strict = NULL;
        s->object_hook = NULL;
        s->object_pairs_hook = NULL;
        s->parse_float = NULL;
        s->parse_int = NULL;
        s->parse_constant = NULL;
    }
    return (PyObject *)s;
}

static int
scanner_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    /* Initialize Scanner object */
    PyObject *ctx;
    static char *kwlist[] = {"context", NULL};
    PyScannerObject *s;

    assert(PyScanner_Check(self));
    s = (PyScannerObject *)self;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O:make_scanner", kwlist, &ctx))
        return -1;

    if (s->memo == NULL) {
        s->memo = PyDict_New();
        if (s->memo == NULL)
            goto bail;
    }

    /* All of these will fail "gracefully" so we don't need to verify them */
    s->strict = PyObject_GetAttrString(ctx, "strict");
    if (s->strict == NULL)
        goto bail;
    s->object_hook = PyObject_GetAttrString(ctx, "object_hook");
    if (s->object_hook == NULL)
        goto bail;
    s->object_pairs_hook = PyObject_GetAttrString(ctx, "object_pairs_hook");
    if (s->object_pairs_hook == NULL)
        goto bail;
    s->parse_float = PyObject_GetAttrString(ctx, "parse_float");
    if (s->parse_float == NULL)
        goto bail;
    s->parse_int = PyObject_GetAttrString(ctx, "parse_int");
    if (s->parse_int == NULL)
        goto bail;
    s->parse_constant = PyObject_GetAttrString(ctx, "parse_constant");
    if (s->parse_constant == NULL)
        goto bail;

    return 0;

bail:
    Py_CLEAR(s->strict);
    Py_CLEAR(s->object_hook);
    Py_CLEAR(s->object_pairs_hook);
    Py_CLEAR(s->parse_float);
    Py_CLEAR(s->parse_int);
    Py_CLEAR(s->parse_constant);
    return -1;
}

PyDoc_STRVAR(scanner_doc, "JSON scanner object");

static
PyTypeObject PyScannerType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_json.Scanner",       /* tp_name */
    sizeof(PyScannerObject), /* tp_basicsize */
    0,                    /* tp_itemsize */
    scanner_dealloc, /* tp_dealloc */
    0,                    /* tp_print */
    0,                    /* tp_getattr */
    0,                    /* tp_setattr */
    0,                    /* tp_compare */
    0,                    /* tp_repr */
    0,                    /* tp_as_number */
    0,                    /* tp_as_sequence */
    0,                    /* tp_as_mapping */
    0,                    /* tp_hash */
    scanner_call,         /* tp_call */
    0,                    /* tp_str */
    0,/* PyObject_GenericGetAttr, */                    /* tp_getattro */
    0,/* PyObject_GenericSetAttr, */                    /* tp_setattro */
    0,                    /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,   /* tp_flags */
    scanner_doc,          /* tp_doc */
    scanner_traverse,                    /* tp_traverse */
    scanner_clear,                    /* tp_clear */
    0,                    /* tp_richcompare */
    0,                    /* tp_weaklistoffset */
    0,                    /* tp_iter */
    0,                    /* tp_iternext */
    0,                    /* tp_methods */
    scanner_members,                    /* tp_members */
    0,                    /* tp_getset */
    0,                    /* tp_base */
    0,                    /* tp_dict */
    0,                    /* tp_descr_get */
    0,                    /* tp_descr_set */
    0,                    /* tp_dictoffset */
    scanner_init,                    /* tp_init */
    0,/* PyType_GenericAlloc, */        /* tp_alloc */
    scanner_new,          /* tp_new */
    0,/* PyObject_GC_Del, */              /* tp_free */
};

static PyObject *
encoder_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyEncoderObject *s;
    s = (PyEncoderObject *)type->tp_alloc(type, 0);
    if (s != NULL) {
        s->markers = NULL;
        s->defaultfn = NULL;
        s->encoder = NULL;
        s->indent = NULL;
        s->key_separator = NULL;
        s->item_separator = NULL;
        s->sort_keys = NULL;
        s->skipkeys = NULL;
    }
    return (PyObject *)s;
}

static int
encoder_init(PyObject *self, PyObject *args, PyObject *kwds)
{
    /* initialize Encoder object */
    static char *kwlist[] = {"markers", "default", "encoder", "indent", "key_separator", "item_separator", "sort_keys", "skipkeys", "allow_nan", NULL};

    PyEncoderObject *s;
    PyObject *markers, *defaultfn, *encoder, *indent, *key_separator;
    PyObject *item_separator, *sort_keys, *skipkeys, *allow_nan;

    assert(PyEncoder_Check(self));
    s = (PyEncoderObject *)self;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOOOOOO:make_encoder", kwlist,
        &markers, &defaultfn, &encoder, &indent, &key_separator, &item_separator,
        &sort_keys, &skipkeys, &allow_nan))
        return -1;

    s->markers = markers;
    s->defaultfn = defaultfn;
    s->encoder = encoder;
    s->indent = indent;
    s->key_separator = key_separator;
    s->item_separator = item_separator;
    s->sort_keys = sort_keys;
    s->skipkeys = skipkeys;
    s->fast_encode = (PyCFunction_Check(s->encoder) && PyCFunction_GetFunction(s->encoder) == (PyCFunction)py_encode_basestring_ascii);
    s->allow_nan = PyObject_IsTrue(allow_nan);

    Py_INCREF(s->markers);
    Py_INCREF(s->defaultfn);
    Py_INCREF(s->encoder);
    Py_INCREF(s->indent);
    Py_INCREF(s->key_separator);
    Py_INCREF(s->item_separator);
    Py_INCREF(s->sort_keys);
    Py_INCREF(s->skipkeys);
    return 0;
}

static PyObject *
encoder_call(PyObject *self, PyObject *args, PyObject *kwds)
{
    /* Python callable interface to encode_listencode_obj */
    static char *kwlist[] = {"obj", "_current_indent_level", NULL};
    PyObject *obj;
    PyObject *rval;
    Py_ssize_t indent_level;
    PyEncoderObject *s;
    assert(PyEncoder_Check(self));
    s = (PyEncoderObject *)self;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO&:_iterencode", kwlist,
        &obj, _convertPyInt_AsSsize_t, &indent_level))
        return NULL;
    rval = PyList_New(0);
    if (rval == NULL)
        return NULL;
    if (encoder_listencode_obj(s, rval, obj, indent_level)) {
        Py_DECREF(rval);
        return NULL;
    }
    return rval;
}

static PyObject *
_encoded_const(PyObject *obj)
{
    /* Return the JSON string representation of None, True, False */
    if (obj == Py_None) {
        static PyObject *s_null = NULL;
        if (s_null == NULL) {
            s_null = PyUnicode_InternFromString("null");
        }
        Py_INCREF(s_null);
        return s_null;
    }
    else if (obj == Py_True) {
        static PyObject *s_true = NULL;
        if (s_true == NULL) {
            s_true = PyUnicode_InternFromString("true");
        }
        Py_INCREF(s_true);
        return s_true;
    }
    else if (obj == Py_False) {
        static PyObject *s_false = NULL;
        if (s_false == NULL) {
            s_false = PyUnicode_InternFromString("false");
        }
        Py_INCREF(s_false);
        return s_false;
    }
    else {
        PyErr_SetString(PyExc_ValueError, "not a const");
        return NULL;
    }
}

static PyObject *
encoder_encode_float(PyEncoderObject *s, PyObject *obj)
{
    /* Return the JSON representation of a PyFloat */
    double i = PyFloat_AS_DOUBLE(obj);
    if (!Py_IS_FINITE(i)) {
        if (!s->allow_nan) {
            PyErr_SetString(PyExc_ValueError, "Out of range float values are not JSON compliant");
            return NULL;
        }
        if (i > 0) {
            return PyUnicode_FromString("Infinity");
        }
        else if (i < 0) {
            return PyUnicode_FromString("-Infinity");
        }
        else {
            return PyUnicode_FromString("NaN");
        }
    }
    /* Use a better float format here? */
    return PyObject_Repr(obj);
}

static PyObject *
encoder_encode_string(PyEncoderObject *s, PyObject *obj)
{
    /* Return the JSON representation of a string */
    if (s->fast_encode)
        return py_encode_basestring_ascii(NULL, obj);
    else
        return PyObject_CallFunctionObjArgs(s->encoder, obj, NULL);
}

static int
_steal_list_append(PyObject *lst, PyObject *stolen)
{
    /* Append stolen and then decrement its reference count */
    int rval = PyList_Append(lst, stolen);
    Py_DECREF(stolen);
    return rval;
}

static int
encoder_listencode_obj(PyEncoderObject *s, PyObject *rval, PyObject *obj, Py_ssize_t indent_level)
{
    /* Encode Python object obj to a JSON term, rval is a PyList */
    PyObject *newobj;
    int rv;

    if (obj == Py_None || obj == Py_True || obj == Py_False) {
        PyObject *cstr = _encoded_const(obj);
        if (cstr == NULL)
            return -1;
        return _steal_list_append(rval, cstr);
    }
    else if (PyUnicode_Check(obj))
    {
        PyObject *encoded = encoder_encode_string(s, obj);
        if (encoded == NULL)
            return -1;
        return _steal_list_append(rval, encoded);
    }
    else if (PyLong_Check(obj)) {
        PyObject *encoded = PyObject_Str(obj);
        if (encoded == NULL)
            return -1;
        return _steal_list_append(rval, encoded);
    }
    else if (PyFloat_Check(obj)) {
        PyObject *encoded = encoder_encode_float(s, obj);
        if (encoded == NULL)
            return -1;
        return _steal_list_append(rval, encoded);
    }
    else if (PyList_Check(obj) || PyTuple_Check(obj)) {
        return encoder_listencode_list(s, rval, obj, indent_level);
    }
    else if (PyDict_Check(obj)) {
        return encoder_listencode_dict(s, rval, obj, indent_level);
    }
    else {
        PyObject *ident = NULL;
        if (s->markers != Py_None) {
            int has_key;
            ident = PyLong_FromVoidPtr(obj);
            if (ident == NULL)
                return -1;
            has_key = PyDict_Contains(s->markers, ident);
            if (has_key) {
                if (has_key != -1)
                    PyErr_SetString(PyExc_ValueError, "Circular reference detected");
                Py_DECREF(ident);
                return -1;
            }
            if (PyDict_SetItem(s->markers, ident, obj)) {
                Py_DECREF(ident);
                return -1;
            }
        }
        newobj = PyObject_CallFunctionObjArgs(s->defaultfn, obj, NULL);
        if (newobj == NULL) {
            Py_XDECREF(ident);
            return -1;
        }
        rv = encoder_listencode_obj(s, rval, newobj, indent_level);
        Py_DECREF(newobj);
        if (rv) {
            Py_XDECREF(ident);
            return -1;
        }
        if (ident != NULL) {
            if (PyDict_DelItem(s->markers, ident)) {
                Py_XDECREF(ident);
                return -1;
            }
            Py_XDECREF(ident);
        }
        return rv;
    }
}

static int
encoder_listencode_dict(PyEncoderObject *s, PyObject *rval, PyObject *dct, Py_ssize_t indent_level)
{
    /* Encode Python dict dct a JSON term, rval is a PyList */
    static PyObject *open_dict = NULL;
    static PyObject *close_dict = NULL;
    static PyObject *empty_dict = NULL;
    PyObject *kstr = NULL;
    PyObject *ident = NULL;
    PyObject *it = NULL;
    PyObject *items;
    PyObject *item = NULL;
    int skipkeys;
    Py_ssize_t idx;

    if (open_dict == NULL || close_dict == NULL || empty_dict == NULL) {
        open_dict = PyUnicode_InternFromString("{");
        close_dict = PyUnicode_InternFromString("}");
        empty_dict = PyUnicode_InternFromString("{}");
        if (open_dict == NULL || close_dict == NULL || empty_dict == NULL)
            return -1;
    }
    if (Py_SIZE(dct) == 0)
        return PyList_Append(rval, empty_dict);

    if (s->markers != Py_None) {
        int has_key;
        ident = PyLong_FromVoidPtr(dct);
        if (ident == NULL)
            goto bail;
        has_key = PyDict_Contains(s->markers, ident);
        if (has_key) {
            if (has_key != -1)
                PyErr_SetString(PyExc_ValueError, "Circular reference detected");
            goto bail;
        }
        if (PyDict_SetItem(s->markers, ident, dct)) {
            goto bail;
        }
    }

    if (PyList_Append(rval, open_dict))
        goto bail;

    if (s->indent != Py_None) {
        /* TODO: DOES NOT RUN */
        indent_level += 1;
        /*
            newline_indent = '\n' + (' ' * (_indent * _current_indent_level))
            separator = _item_separator + newline_indent
            buf += newline_indent
        */
    }

    if (PyObject_IsTrue(s->sort_keys)) {
        /* First sort the keys then replace them with (key, value) tuples. */
        Py_ssize_t i, nitems;
        items = PyMapping_Keys(dct);
        if (items == NULL)
            goto bail;
        if (!PyList_Check(items)) {
            PyErr_SetString(PyExc_ValueError, "keys must return list");
            goto bail;
        }
        if (PyList_Sort(items) < 0)
            goto bail;
        nitems = PyList_GET_SIZE(items);
        for (i = 0; i < nitems; i++) {
            PyObject *key, *value;
            key = PyList_GET_ITEM(items, i);
            value = PyDict_GetItem(dct, key);
            item = PyTuple_Pack(2, key, value);
            if (item == NULL)
                goto bail;
            PyList_SET_ITEM(items, i, item);
            Py_DECREF(key);
        }
    }
    else {
        items = PyMapping_Items(dct);
    }
    if (items == NULL)
        goto bail;
    it = PyObject_GetIter(items);
    Py_DECREF(items);
    if (it == NULL)
        goto bail;
    skipkeys = PyObject_IsTrue(s->skipkeys);
    idx = 0;
    while ((item = PyIter_Next(it)) != NULL) {
        PyObject *encoded, *key, *value;
        if (!PyTuple_Check(item) || Py_SIZE(item) != 2) {
            PyErr_SetString(PyExc_ValueError, "items must return 2-tuples");
            goto bail;
        }
        key = PyTuple_GET_ITEM(item, 0);
        if (PyUnicode_Check(key)) {
            Py_INCREF(key);
            kstr = key;
        }
        else if (PyFloat_Check(key)) {
            kstr = encoder_encode_float(s, key);
            if (kstr == NULL)
                goto bail;
        }
        else if (key == Py_True || key == Py_False || key == Py_None) {
                        /* This must come before the PyLong_Check because
                           True and False are also 1 and 0.*/
            kstr = _encoded_const(key);
            if (kstr == NULL)
                goto bail;
        }
        else if (PyLong_Check(key)) {
            kstr = PyObject_Str(key);
            if (kstr == NULL)
                goto bail;
        }
        else if (skipkeys) {
            Py_DECREF(item);
            continue;
        }
        else {
            /* TODO: include repr of key */
            PyErr_SetString(PyExc_TypeError, "keys must be a string");
            goto bail;
        }

        if (idx) {
            if (PyList_Append(rval, s->item_separator))
                goto bail;
        }

        encoded = encoder_encode_string(s, kstr);
        Py_CLEAR(kstr);
        if (encoded == NULL)
            goto bail;
        if (PyList_Append(rval, encoded)) {
            Py_DECREF(encoded);
            goto bail;
        }
        Py_DECREF(encoded);
        if (PyList_Append(rval, s->key_separator))
            goto bail;

        value = PyTuple_GET_ITEM(item, 1);
        if (encoder_listencode_obj(s, rval, value, indent_level))
            goto bail;
        idx += 1;
        Py_DECREF(item);
    }
    if (PyErr_Occurred())
        goto bail;
    Py_CLEAR(it);

    if (ident != NULL) {
        if (PyDict_DelItem(s->markers, ident))
            goto bail;
        Py_CLEAR(ident);
    }
    /* TODO DOES NOT RUN; dead code
    if (s->indent != Py_None) {
        indent_level -= 1;

        yield '\n' + (' ' * (_indent * _current_indent_level))
    }*/
    if (PyList_Append(rval, close_dict))
        goto bail;
    return 0;

bail:
    Py_XDECREF(it);
    Py_XDECREF(item);
    Py_XDECREF(kstr);
    Py_XDECREF(ident);
    return -1;
}


static int
encoder_listencode_list(PyEncoderObject *s, PyObject *rval, PyObject *seq, Py_ssize_t indent_level)
{
    /* Encode Python list seq to a JSON term, rval is a PyList */
    static PyObject *open_array = NULL;
    static PyObject *close_array = NULL;
    static PyObject *empty_array = NULL;
    PyObject *ident = NULL;
    PyObject *s_fast = NULL;
    Py_ssize_t num_items;
    PyObject **seq_items;
    Py_ssize_t i;

    if (open_array == NULL || close_array == NULL || empty_array == NULL) {
        open_array = PyUnicode_InternFromString("[");
        close_array = PyUnicode_InternFromString("]");
        empty_array = PyUnicode_InternFromString("[]");
        if (open_array == NULL || close_array == NULL || empty_array == NULL)
            return -1;
    }
    ident = NULL;
    s_fast = PySequence_Fast(seq, "_iterencode_list needs a sequence");
    if (s_fast == NULL)
        return -1;
    num_items = PySequence_Fast_GET_SIZE(s_fast);
    if (num_items == 0) {
        Py_DECREF(s_fast);
        return PyList_Append(rval, empty_array);
    }

    if (s->markers != Py_None) {
        int has_key;
        ident = PyLong_FromVoidPtr(seq);
        if (ident == NULL)
            goto bail;
        has_key = PyDict_Contains(s->markers, ident);
        if (has_key) {
            if (has_key != -1)
                PyErr_SetString(PyExc_ValueError, "Circular reference detected");
            goto bail;
        }
        if (PyDict_SetItem(s->markers, ident, seq)) {
            goto bail;
        }
    }

    seq_items = PySequence_Fast_ITEMS(s_fast);
    if (PyList_Append(rval, open_array))
        goto bail;
    if (s->indent != Py_None) {
        /* TODO: DOES NOT RUN */
        indent_level += 1;
        /*
            newline_indent = '\n' + (' ' * (_indent * _current_indent_level))
            separator = _item_separator + newline_indent
            buf += newline_indent
        */
    }
    for (i = 0; i < num_items; i++) {
        PyObject *obj = seq_items[i];
        if (i) {
            if (PyList_Append(rval, s->item_separator))
                goto bail;
        }
        if (encoder_listencode_obj(s, rval, obj, indent_level))
            goto bail;
    }
    if (ident != NULL) {
        if (PyDict_DelItem(s->markers, ident))
            goto bail;
        Py_CLEAR(ident);
    }

    /* TODO: DOES NOT RUN
    if (s->indent != Py_None) {
        indent_level -= 1;

        yield '\n' + (' ' * (_indent * _current_indent_level))
    }*/
    if (PyList_Append(rval, close_array))
        goto bail;
    Py_DECREF(s_fast);
    return 0;

bail:
    Py_XDECREF(ident);
    Py_DECREF(s_fast);
    return -1;
}

static void
encoder_dealloc(PyObject *self)
{
    /* Deallocate Encoder */
    encoder_clear(self);
    Py_TYPE(self)->tp_free(self);
}

static int
encoder_traverse(PyObject *self, visitproc visit, void *arg)
{
    PyEncoderObject *s;
    assert(PyEncoder_Check(self));
    s = (PyEncoderObject *)self;
    Py_VISIT(s->markers);
    Py_VISIT(s->defaultfn);
    Py_VISIT(s->encoder);
    Py_VISIT(s->indent);
    Py_VISIT(s->key_separator);
    Py_VISIT(s->item_separator);
    Py_VISIT(s->sort_keys);
    Py_VISIT(s->skipkeys);
    return 0;
}

static int
encoder_clear(PyObject *self)
{
    /* Deallocate Encoder */
    PyEncoderObject *s;
    assert(PyEncoder_Check(self));
    s = (PyEncoderObject *)self;
    Py_CLEAR(s->markers);
    Py_CLEAR(s->defaultfn);
    Py_CLEAR(s->encoder);
    Py_CLEAR(s->indent);
    Py_CLEAR(s->key_separator);
    Py_CLEAR(s->item_separator);
    Py_CLEAR(s->sort_keys);
    Py_CLEAR(s->skipkeys);
    return 0;
}

PyDoc_STRVAR(encoder_doc, "_iterencode(obj, _current_indent_level) -> iterable");

static
PyTypeObject PyEncoderType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_json.Encoder",       /* tp_name */
    sizeof(PyEncoderObject), /* tp_basicsize */
    0,                    /* tp_itemsize */
    encoder_dealloc, /* tp_dealloc */
    0,                    /* tp_print */
    0,                    /* tp_getattr */
    0,                    /* tp_setattr */
    0,                    /* tp_compare */
    0,                    /* tp_repr */
    0,                    /* tp_as_number */
    0,                    /* tp_as_sequence */
    0,                    /* tp_as_mapping */
    0,                    /* tp_hash */
    encoder_call,         /* tp_call */
    0,                    /* tp_str */
    0,                    /* tp_getattro */
    0,                    /* tp_setattro */
    0,                    /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,   /* tp_flags */
    encoder_doc,          /* tp_doc */
    encoder_traverse,     /* tp_traverse */
    encoder_clear,        /* tp_clear */
    0,                    /* tp_richcompare */
    0,                    /* tp_weaklistoffset */
    0,                    /* tp_iter */
    0,                    /* tp_iternext */
    0,                    /* tp_methods */
    encoder_members,      /* tp_members */
    0,                    /* tp_getset */
    0,                    /* tp_base */
    0,                    /* tp_dict */
    0,                    /* tp_descr_get */
    0,                    /* tp_descr_set */
    0,                    /* tp_dictoffset */
    encoder_init,         /* tp_init */
    0,                    /* tp_alloc */
    encoder_new,          /* tp_new */
    0,                    /* tp_free */
};

static PyMethodDef speedups_methods[] = {
    {"encode_basestring_ascii",
        (PyCFunction)py_encode_basestring_ascii,
        METH_O,
        pydoc_encode_basestring_ascii},
    {"scanstring",
        (PyCFunction)py_scanstring,
        METH_VARARGS,
        pydoc_scanstring},
    {NULL, NULL, 0, NULL}
};

PyDoc_STRVAR(module_doc,
"json speedups\n");

static struct PyModuleDef jsonmodule = {
        PyModuleDef_HEAD_INIT,
        "_json",
        module_doc,
        -1,
        speedups_methods,
        NULL,
        NULL,
        NULL,
        NULL
};

PyObject*
PyInit__json(void)
{
    PyObject *m = PyModule_Create(&jsonmodule);
    if (!m)
        return NULL;
    PyScannerType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&PyScannerType) < 0)
        goto fail;
    PyEncoderType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&PyEncoderType) < 0)
        goto fail;
    Py_INCREF((PyObject*)&PyScannerType);
    if (PyModule_AddObject(m, "make_scanner", (PyObject*)&PyScannerType) < 0) {
        Py_DECREF((PyObject*)&PyScannerType);
        goto fail;
    }
    Py_INCREF((PyObject*)&PyEncoderType);
    if (PyModule_AddObject(m, "make_encoder", (PyObject*)&PyEncoderType) < 0) {
        Py_DECREF((PyObject*)&PyEncoderType);
        goto fail;
    }
    return m;
  fail:
    Py_DECREF(m);
    return NULL;
}
