    protected void writeDataDescriptor(final ZipArchiveEntry ze) throws IOException {
        if (ze.getMethod() != DEFLATED || channel != null) {
            return;
        }
        writeCounted(DD_SIG);
        writeCounted(ZipLong.getBytes(ze.getCrc()));
        if (!hasZip64Extra(ze)) {
            writeCounted(ZipLong.getBytes(ze.getCompressedSize()));
            writeCounted(ZipLong.getBytes(ze.getSize()));
        } else {
            writeCounted(ZipEightByteInteger.getBytes(ze.getCompressedSize()));
            writeCounted(ZipEightByteInteger.getBytes(ze.getSize()));
        }
    }