        protected SortedMap<K, V> createRangeMap(final K fromKey, final boolean fromInclusive,
                                                 final K toKey, final boolean toInclusive) {
            return new RangeEntryMap(fromKey, fromInclusive, toKey, toInclusive);
        }